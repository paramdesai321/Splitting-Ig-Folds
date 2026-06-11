import sys
import re
arg1 = sys.argv[1]
#arg2 = sys.argv[2]
#arg3 = sys.argv[3]
def merge_pdb_files_simple(file1, file2, output_file):
    with open(output_file, 'w') as outfile:
        for pdb_file in [file1, file2]:
            with open(pdb_file, 'r') as infile:
                for line in infile:
                    if not line.startswith("END"):  # Skip END lines to avoid premature termination
                        outfile.write(line)
        outfile.write("END\n")  # Write one final END line

# Example usage

#arg2 = re.add(r"_seg\d+$","",arg1)
arg2 = arg1 + "_seg0"
print(arg1)
file1 = f"./aligned_pdbs_CD_HIT_90/aligned_{arg2}.pdb"
#file2 = f"./Plane_CD_HIT_90/plane_{arg2}.pdb"
file2 = f"./Plane_CD_HIT_90/plane_{arg2}.pdb"
merge_pdb_files_simple(file1, file2,f"./final_transformation_CD_HIT_90/{arg2}_withPlane.pdb")

