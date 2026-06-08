#!/usr/bin/env python3
from pathlib import Path
from pprint import pprint
import os
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import sys 

pdb_file = sys.argv[1]


def detect_strands_to_dict(pdb_path, chain_id=None, min_len=1, dssp_exec="mkdssp"):
    """
    Returns:
    {
        PDB_NAME: {
            "all_residues": [list of all strand residues],
            "strands": {
                1: [residues in strand 1],
                2: [residues in strand 2],
                ...
            }
        }
    }
    """

    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = next(structure.get_models())

    dssp = DSSP(model, pdb_path, dssp=dssp_exec)

    # Collect strand residues per chain
    strand_residues = []

    for key in dssp.keys():
        chain, res_id = key
        if chain_id and chain != chain_id:
            continue

        ss = dssp[key][2]  # secondary structure code
        if ss == "E":  # beta strand
            resseq = res_id[1]
            strand_residues.append(resseq)

    strand_residues = sorted(strand_residues)


    strands = {}
    strand_index = 1

    if strand_residues:
        current = [strand_residues[0]]
        for r in strand_residues[1:]:
            gap = r - current[-1]

            # Merge strands separated by at most one missing residue
            if gap <= 1:
                current.append(r)
            else:
                if len(current) >= min_len:
                    strands[strand_index] = current
                    strand_index += 1
                current = [r]
       # final strand
        if len(current) >= min_len:
            strands[strand_index] = current

    # Flatten all residues from valid strands
    all_res = []
    for s in strands.values():
        all_res.extend(s)

    result = {
        pdb_name: {
            "all_residues": all_res,
            "strands": strands
        }
    }

    return result
#pdb_file = "../output_pdbs/1A4K_L_3_107.pdb"
#pdb_file = "../output_pdbs/4PB0_L_2_107.pdb"
#pdb_file = "../output_pdbs/1YJD_C_3_117.pdb"
chain_id = Path(pdb_file).stem.split("_")[1]
result = detect_strands_to_dict(pdb_file, chain_id=chain_id, min_len=3)
print(result)
result_dict = result
if __name__ == "__main__":
    pdb_file = "../output_pdbs/1A4K_L_3_107.pdb"
    #pdb_file = "../output_pdbs/4PB0_L_2_107.pdb"
    pdb_file= "../output_pdbs/1YJD_C_3_117.pdb"
    result = detect_strands_to_dict(pdb_file, chain_id="L", min_len=3)
    print(result)
