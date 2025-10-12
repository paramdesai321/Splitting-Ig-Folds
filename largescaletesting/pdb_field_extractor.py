from parsing_coords import get_atom_lines

def extracted_dict(pdb_lines):
    line_list= []
    for line in pdb_lines:
        if line.startswith("ATOM"):
            properties = {
                "record": line[0:6].strip(),
                "atom_serial": int(line[6:11].strip()),
                "atom_name": line[12:16].strip(),
                "alt_loc": line[16:17].strip(),
                "res_name": line[17:20].strip(),
                "chain_id": line[21:22].strip(),
#                "chain_id": 'X',
                "res_seq": int(line[22:26].strip()),
                "insertion_code": line[26:27].strip(),
                "x": float(line[30:38].strip()),
                "y": float(line[38:46].strip()),
                "z": float(line[46:54].strip()),
                # --- GEMINI FIX: Handle potentially empty optional fields ---
                # The occupancy and temp_factor fields can be blank in PDB files.
                # This checks if the field is empty and defaults to 0.0 if it is.
                "occupancy": float(line[54:60].strip()) if line[54:60].strip() else 0.0,
                "temp_factor": float(line[60:66].strip()) if line[60:66].strip() else 0.0,
                # --- Original lines commented out by Gemini ---
                # "occupancy": float(line[54:60].strip()),
                # "temp_factor":float(line[60:66].strip()),
                "element": line[76:78].strip(),
                "charge": line[78:80].strip(),
            }
            line_list.append(properties)
    return line_list


#print(len(test))
