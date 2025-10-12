import os
import sys
from sklearn_svm import apply_rotation_matrix 
from parsing_coords import coordinates
import numpy as np

coords_from_protien = coordinates()

coords_from_protein = apply_rotation_matrix(coords_from_protien) # The transpose will make the matrix (x,3) 
coords_from_protein = np.transpose(apply_rotation_matrix(coords_from_protien)) # The transpose will make the matrix (x,3) 


def orient(read_file,write_file):
    with open(read_file,'r') as rf:
        with open(write_file,'w') as wf:
            atom_i = 0
            for line in rf:
                if line.startswith(('ATOM  ', 'HETATM')):
                    if atom_i >= len(coords_from_protein):
                        raise ValueError(f"Too few coords: {len(coords)} provided for >{atom_i} atoms")
                    x, y, z = coords_from_protein[atom_i]
                    # PDB fixed-width: columns 1-30, 31-38 X, 39-46 Y, 47-54 Z, 55+ rest
                    new_line = (
                        f"{line[:30]}"
                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                        f"{line[54:]}"
                    )
                    wf.write(new_line)
                    atom_i += 1
                else:
                    wf.write(line)
 

PIN = sys.argv[1]
orient(f'{PIN}.pdb',f'oriented{PIN}.pdb')
orient(f'./dir_BCEF/{PIN}_BCEF.pdb',f'oriented{PIN}_BCEF.pdb')
   
