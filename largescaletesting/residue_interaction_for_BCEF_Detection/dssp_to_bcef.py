
from pathlib import Path
from collections import Counter
import sys

from detect_strands_to_dict import detect_strands_to_dict


def parse_bcef_sheet_ranges(bcef_pdb_path):
    """
    Parse BCEF-labeled SHEET records.

    Returns:
        [
            {"label": "B", "start": 15, "end": 26},
            {"label": "C", "start": 30, "end": 38},
            ...
        ]
    """
    ranges = []

    with open(bcef_pdb_path, "r") as f:
        for line in f:
            if not line.startswith("SHEET"):
                continue

            label = line[21].strip()
            start = int(line[22:26].strip())
            end = int(line[33:37].strip())

            ranges.append({
                "label": label,
                "start": start,
                "end": end
            })

    return ranges


def label_dssp_strands(dssp_result, bcef_ranges):
    """
    Convert DSSP strand dict to strand -> BCEF label.

    dssp_result format:
        {
            pdb_name: {
                "all_residues": [...],
                "strands": {
                    1: [...],
                    2: [...],
                    ...
                }
            }
        }

    Returns:
        {
            1: None,
            2: "B",
            3: "C",
            ...
        }
    """
    pdb_name = next(iter(dssp_result))
    strands = dssp_result[pdb_name]["strands"]

    strand_labels = {}

    for strand_id, residues in strands.items():
        votes = []

        for r in residues:
            for rg in bcef_ranges:
                if rg["start"] <= r <= rg["end"]:
                    votes.append(rg["label"])

        strand_labels[strand_id] = (
            Counter(votes).most_common(1)[0][0]
            if votes else None
        )

    return strand_labels


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python label_dssp_with_bcef.py <normal_pdb_file> <bcef_pdb_file>")
        sys.exit(1)

    pdb_file = sys.argv[1]
    bcef_file = sys.argv[2]

    chain_id = Path(pdb_file).stem.split("_")[1]

    dssp_result = detect_strands_to_dict(
        pdb_file,
        chain_id=chain_id,
        min_len=3
    )

    bcef_ranges = parse_bcef_sheet_ranges(bcef_file)

    labeled = label_dssp_strands(
        dssp_result=dssp_result,
        bcef_ranges=bcef_ranges
    )

    print(labeled)


if __name__ == "__main__":
    main()
