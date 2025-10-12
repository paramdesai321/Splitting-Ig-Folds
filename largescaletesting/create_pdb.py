import sys
import numpy as np
import parsing_coords 
import y_alignment
import pdb_field_extractor
import parsing_BCEF
PIN = sys.argv[1]
atom_line = parsing_coords.line_per_file(PIN)
#print(atom_line)
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

def seperate_BCEF(atom_dict):
    map_dict = parsing_BCEF.res_to_chain_map(PIN)
    res_id   = atom_dict["res_seq"]

    if res_id in map_dict:
        atom_dict['chain_id'] = map_dict[res_id]
    else:
         atom_dict['chain_id'] = 'X'
      
def create_pdb_with_coordinates(coords=None,file_type='custom'):
    output_file  = f'./aligned_pdbs/{file_type}.pdb'
    with open(output_file, "w") as f:
        for i in range(len(coords[:,0])):
            
            atom_list = pdb_field_extractor.extracted_dict(atom_line)
            atom_dict = atom_list[i]
  #          print(f"Atom dict: {atom_dict}")
            seperate_BCEF(atom_dict)
 #           print(f"Atom dict : {atom_dict}")
            
            line = (
                "{record:6s}{atom_serial:5d} {atom_name:^4s}{alt_loc:1s}"
                "{res_name:3s} {chain_id:1s}{res_seq:4d}{insertion_code:1s}   "
                "{0:8.3f}{1:8.3f}{2:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          "
                "{element:>2s}{charge:2s}"
            ).format(coords[i][0], coords[i][1], coords[i][2], **atom_dict)

            f.write(line + "\n")
#print(y_alignment.y_aligned_protein_BCEF.T.shape)
create_pdb_with_coordinates(coords = y_alignment.y_aligned_protein,file_type = f'final_{PIN}')
#    
#create_pdb_with_coordinates(coords = y_alignment.y_aligned_protein,file_type = 'final_Protein')

