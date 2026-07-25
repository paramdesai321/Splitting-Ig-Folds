#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import re
import os
import detect_strands_to_dict
from collections import defaultdict
import sys

pdb_path= sys.argv[1]
BACKBONE_ATOMS = {"N", "CA", "C"}
def parse_pdb_backbone_coords_by_strand(pdb_path: str, chain_id: str, strands: dict[int, list[int]],BACKBONE_ATOMS={"N","CA","C"}):
    """
    Returns:
      strand_coords[strand_idx] = [[x,y,z], [x,y,z], ...]  # backbone atoms across residues in that strand
    Only includes ATOM lines for the given chain, residue numbers in each strand list, and atoms N/CA/C.
    """

    # Precompute fast membership sets per strand
    strand_sets = {k: set(v) for k, v in strands.items()}

    # Output dict: strand -> list of coords
    strand_coords = {k: [] for k in strands.keys()}

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name not in BACKBONE_ATOMS:
                continue

            line_chain = line[21].strip()
            if line_chain != chain_id:
                continue

            # Residue sequence number is columns 22-26 in PDB format
            # (strip handles spaces; int() handles "  45")
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue

            # Parse coordinates
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue

            # Assign to whichever strand contains this residue
            # (Residues shouldn't belong to multiple strands, but we handle safely.)
            for strand_idx, resset in strand_sets.items():
                if resseq in resset:
                    strand_coords[strand_idx].append([x, y, z])
                    break

    return strand_coords


def parse_pdb_backbone_coords_by_strand_and_residue(pdb_path: str, chain_id: str, strands: dict[int, list[int]]):
    """
    Slightly richer structure (often more useful):

      out[strand_idx][resseq][atom_name] = [x,y,z]

    This preserves which residue/atom each coordinate came from.
    """
    strand_sets = {k: set(v) for k, v in strands.items()}
    out = {k: defaultdict(dict) for k in strands.keys()}

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name not in BACKBONE_ATOMS:
                continue

            if line[21].strip() != chain_id:
                continue

            try:
                resseq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue

            for strand_idx, resset in strand_sets.items():
                if resseq in resset:
                    out[strand_idx][resseq][atom_name] = [x, y, z]
                    break

    # convert inner defaultdicts to normal dicts for cleanliness
    return {s: dict(resmap) for s, resmap in out.items()}

#def parse_pdb_backbone_coords_by_strand(pdb_path,chain_id):
#    strands = detect_strands_to_dict.result_dict[pdb_name]['strands']
#    return parse_pdb_backbone_coords_by_strands(pdb_path,chain_id,strands)

def centroid_per_strand_dict(pdb_path,chain_id):
    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
    strands = detect_strands_to_dict.detect_strands_to_dict(
        pdb_path,
        chain_id=chain_id,
        min_len=3
        )[pdb_name]['strands']
    strand_coords = parse_pdb_backbone_coords_by_strand(pdb_path,chain_id,strands)
    centroid = {}
    for strand in strand_coords.keys():
        centroid[strand] = np.array(strand_coords[strand])
        centroid[strand] = np.mean(strand_coords[strand],axis=0)
    return centroid
        
 
    
if __name__ == "__main__":
    # Your example strands dict
   # pdb_path = "../output_pdbs/4PB0_L_2_107.pdb"
    
    #pdb_path = "../output_pdbs/1A4K_L_3_107.pdb"
    #pdb_path = "../output_pdbs/4PB0_L_2_107.pdb"
    #pdb_path = "../output_pdbs/1YJD_C_3_117.pdb"
    chain_id = Path(pdb_path).stem.split("_")[1]
    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
    strands = detect_strands_to_dict.detect_strands_to_dict(
        pdb_path,
        chain_id=chain_id,
        min_len=3
        )[pdb_name]['strands']
    strand_coords = parse_pdb_backbone_coords_by_strand(pdb_path, chain_id, strands)
    print(strand_coords)
    print("Per-strand counts (N/CA/C atoms total):")
    for k in sorted(strand_coords):
        print(k, len(strand_coords[k]))

    # If you want residue/atom mapping:
    # strand_map = parse_pdb_backbone_coords_by_strand_and_residue(pdb_path, chain_id, strands)
    # print(strand_map[1][4]["CA"])
    #print(f'Centroid : {centroid_per_strand_dict(pdb_path,'C')}')

