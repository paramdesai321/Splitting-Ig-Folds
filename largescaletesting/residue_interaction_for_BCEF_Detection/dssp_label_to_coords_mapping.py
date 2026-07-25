import dssp_label_to_range_mapping 
from pathlib import Path

import sys
import os

def attach_backbone_coords_by_label(pdb_path, chain_id, label_mapping):
    """
    Parameters
    ----------
    pdb_path : str
    chain_id : str
    label_mapping :
    {
        "A'": {"strand_id":1, "start":4, "end":6, "residues":[4,5,6]},
        "A":  {"strand_id":2, "start":10, "end":13, "residues":[10,11,12,13]},
        ...
    }

    Returns
    -------
    {
        "A'": {
            "strand_id":1,
            "start":4,
            "end":6,
            "residues":[4,5,6],
            "coords":[
                [x,y,z],   # N4
                [x,y,z],   # CA4
                [x,y,z],   # C4
                [x,y,z],   # N5
                ...
            ]
        },
        ...
    }
    """

    backbone = {"N", "CA", "C"}

    # Make a copy so we don't modify the input dictionary
    updated = {
        label: {
            **info,
            "coords": []
        }
        for label, info in label_mapping.items()
    }

    with open(pdb_path, "r") as f:
        for line in f:

            if not line.startswith("ATOM"):
                continue

            if line[21].strip() != chain_id:
                continue

            atom = line[12:16].strip()
            if atom not in backbone:
                continue

            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue

            coord = [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]

            for label, info in updated.items():
                if resseq in info["residues"]:
                    info["coords"].append(coord)
                    break

    return updated
pdb_path= sys.argv[1]
chain_id = Path(pdb_path).stem.split("_")[1]
pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
label_mapping = dssp_label_to_range_mapping.ranges
print(label_mapping)
backbone = attach_backbone_coords_by_label(pdb_path, chain_id, label_mapping)
#print(backbone)
