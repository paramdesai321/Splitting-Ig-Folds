import sys
import numpy as np
from parsing_coords import get_atom_lines
import y_alignment
import pdb_field_extractor
PIN = sys.argv[1]
def format_pdb_atom_line(atom_dict):
    return (
        "{record:6s}"                                # 1-6   "ATOM  "
        "{atom_serial:5d} "                           # 7-11  Atom serial
        "{atom_name:^4s}"                              # 13-16 Atom name (right-aligned)
        "{alt_loc:1s}"                                 # 17    Alt loc
        "{res_name:>3s} "                              # 18-20 Residue name
        "{chain_id:1s}"                                # 22    Chain ID
        "{res_seq:4d}"                                # 23-26 Residue seq
        "{insertion_code:1s}   "                       # 27    Insertion code
        "{x:8.3f}"                                    # 31-38 X
        "{y:8.3f}"                                    # 39-46 Y
        "{z:8.3f}"                                    # 47-54 Z
        "{occupancy:6.2f}"                            # 55-60 Occupancy
        "{temp_factor:6.2f}"                          # 61-66 Temp factor
        "          "                                  # 67-76 (padding)
        "{element:>2s}"                                # 77-78 Element
        "{charge:2s}"                                 # 79-80 Charge
    ).format(**atom_dict)
def create_pdb_with_coordinates(coords=None,file_type='custom'):
    output_file  = f'{file_type}.pdb'
    atom_lines = get_atom_lines()
    with open(output_file, "w") as f:
        for i in range(len(coords[:,0])):
            
            atom_list = pdb_field_extractor.extracted_dict(get_atom_lines())
            atom_dict = atom_list[i]
            
            line = (
                "{record:6s}{atom_serial:5d} {atom_name:^4s}{alt_loc:1s}"
                "{res_name:3s} {chain_id:1s}{res_seq:4d}{insertion_code:1s}   "
                "{0:8.3f}{1:8.3f}{2:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          "
                "{element:>2s}{charge:2s}"
            ).format(coords[i][0], coords[i][1], coords[i][2], **atom_dict)

            f.write(line + "\n")
print(y_alignment.y_aligned_protein_BCEF.T.shape)
create_pdb_with_coordinates(coords = y_alignment.y_aligned_protein.T,file_type = f'final_{PIN}')


#create_pdb_with_coordinates(coords = y_alignment.y_aligned_protein,file_type = 'final_Protein')

