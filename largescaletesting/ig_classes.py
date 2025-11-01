import feature_vector
import color_igtype
import main
super_dict = main.super_dict

pdb_by_ig = color_igtype.pdb_by_ig
#print(pdb_by_ig)

for struct_id in super_dict.keys():
    # extract the PDB ID (before the underscore, e.g., "3UMN" from "3UMN_A_435_to_547")
    pdb_id = struct_id
    #print(pdb_id)
    
    # find which Ig type this PDB belongs to
    for ig_type, pdb_list in pdb_by_ig.items():
        #print(ig_type)
        #print(pdb_list)
        if pdb_id in pdb_list:
            super_dict[struct_id]['Type'] = ig_type
            break
    else:
        super_dict[struct_id]['Type'] = 'Unknown'  # optional fallback

ig_type, pdb_list in pdb_by_ig.items()
#print("-------------------")
#print(pdb_list)

#print('3UMN_A_435_to_547' in pdb_list)
#for key in super_dict:
#    print(key, "→", super_dict[key].get("Type"))

