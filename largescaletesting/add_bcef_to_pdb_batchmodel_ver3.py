#!/home/exec/programs/anaconda310/bin/python3

import os
import argparse
import subprocess

def split_pdb_by_bcef(input_file, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, 'r') as pdb_file:
        lines = pdb_file.readlines()

    residue_numbers = []
    atom_lines = []

    for line in lines:
        if line.startswith("ATOM"):
            try:
                res_num = int(line[22:26].strip())
                residue_numbers.append(res_num)
                atom_lines.append(line)

            except ValueError:
                continue


    jumps = []
    for i in range(1, len(residue_numbers)):
        if residue_numbers[i] > residue_numbers[i-1] + 2:
            jumps.append(i)

    itag = 0
    beta_sheet = []
    start = 0
    if len(jumps) >= 4:
        itag = 1
        for j in range(0, len(jumps), 4):
            if j + 3 < len(jumps):
                end = jumps[j + 3]
            else:
                end = len(atom_lines)
            beta_sheet.append(atom_lines[start:end])
            start = end
    else:
        beta_sheet.append(atom_lines)

    split_files = []
    base_name = os.path.basename(input_file).replace(".pdb", "")

    for i, block in enumerate(beta_sheet):
        tag = 'seg'+str(i+1)
        if (itag == 0):
           tag = 'seg0'
        split_file = os.path.join(output_folder, f"{base_name}chains_{tag}.pdb")
        with open(split_file, 'w') as out_file:
            out_file.writelines(block)
        split_files.append(split_file)

    return split_files

def add_bcef_to_pdb(input_folder, output_folder):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bcef_script = os.path.join(script_dir, "ig_bcef_rechain_ver3.py")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".pdb"):
            input_file = os.path.join(input_folder, file)
            split_files = split_pdb_by_bcef(input_file, output_folder)

            for split_file in split_files:
                output_file = split_file.replace(".pdb", "_updated.pdb")
                subprocess.run([bcef_script, "-i", split_file, "-o", output_file])
                os.remove(split_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()
    add_bcef_to_pdb(args.folder, args.output)
