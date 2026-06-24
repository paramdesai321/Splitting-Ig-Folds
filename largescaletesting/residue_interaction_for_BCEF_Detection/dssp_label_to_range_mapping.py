### NOTE: The Residues are only of th backbone atoms!!!!
def get_dssp_ranges_by_label(dssp_result, labels):
    """
    dssp_result:
        {
            pdb_name: {
                "all_residues": [...],
                "strands": {
                    1: [4,5,6],
                    2: [10,11,12,13],
                    ...
                }
            }
        }

    labels:
        {
            1: "A'",
            2: "A",
            3: "B",
            ...
        }

    Returns:
        {
            "A'": {"strand_id": 1, "start": 4, "end": 6, "residues": [4,5,6]},
            "A":  {"strand_id": 2, "start": 10, "end": 13, "residues": [10,11,12,13]},
            ...
        }
    """

    pdb_name = next(iter(dssp_result))
    strands = dssp_result[pdb_name]["strands"]

    ranges_by_label = {}

    for strand_id, label in labels.items():
        if label is None:
            continue

        residues = strands[strand_id]

        ranges_by_label[label] = {
            "strand_id": strand_id,
            "start": min(residues),
            "end": max(residues),
            "residues": residues
        }

    return ranges_by_label
