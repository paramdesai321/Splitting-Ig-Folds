import detect_strands_to_dict
from collections import Counter
import sys

pdb_file = sys.argv[1]
chain_id = Path(pdb_file).stem.split("_")[1]
result = detect_strands_to_dict(pdb_file, chain_id=chain_id, min_len=3)

def parse_bcef_sheet_ranges(bcef_pdb_path):
    ranges = []

    with open(bcef_pdb_path, "r") as f:
        for line in f:
            if not line.startswith("SHEET"):
                continue

            label = line[21].strip()          # B/C/E/F label
            start = int(line[22:26].strip())  # start residue
            end = int(line[33:37].strip())    # end residue

            ranges.append({
                "label": label,
                "start": start,
                "end": end
            })

    return ranges
def label_dssp_strands(dssp_result, bcef_ranges):
    """
    dssp_result:
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
        1: "B",
        2: "C",
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

labeled = label_dssp_result_with_bcef(
    dssp_result=result,
    bcef_pdb_path=pdb_file
)
