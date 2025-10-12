import sys
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
file1 = f"{arg1}.pdb"
file2 = f"plane_{arg1}.pdb"
merge_pdb_files_simple(file1, file2,f"{arg1}withPlane.pdb")

