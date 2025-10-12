#!/home/exec/programs/anaconda310/bin/python

import os

def beta_strand_determination(pdb_file, output_file):
  with open(pdb_file, 'r') as file:
    lines = file.readlines()

  strand_labels = ['B', 'C', 'E', 'F']
  strand_id_dict = {}
  label_index = 0

  for line in lines:
    if line.startswith("SHEET") and label_index < len(strand_labels):
      start_resID = int(line[22:26].strip())
      end_resID = int(line[33:37].strip())
      for resID in range(start_resID, end_resID + 1):
        strand_id_dict[resID] = strand_labels[label_index]
      label_index += 1

  updated_lines = []
  lineCount = 0

  for line in lines:
    if line.startswith("ATOM"):
      lineCount += 1
      resID = int(line[22:26].strip())

      if resID in strand_id_dict:
        new_line = line[:21] + strand_id_dict[resID] + line[22:]
        updated_lines.append(new_line)
      else:
        updated_lines.append(line)
    else:
      updated_lines.append(line)

  with open(output_file, 'w') as file:
    file.writelines(updated_lines)


# main code starts here
if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Conserved beta strand determination for folder')
  parser.add_argument("-i", "--input_folder", required=True, help="Folder with PDB files")
  parser.add_argument("-o", "--output_folder", required=True, help="Folder to write updated files")
  args = parser.parse_args()

  input_folder = args.input_folder
  output_folder = args.output_folder

  os.makedirs(output_folder, exist_ok=True)

  for filename in os.listdir(input_folder):
    if filename.endswith(".pdb"):
      input_path = os.path.join(input_folder, filename)

      # Add "_labeled" suffix before the .pdb extension
      name_part = filename.rsplit(".", 1)[0]
      output_filename = f"{name_part}_labeled.pdb"
      output_path = os.path.join(output_folder, output_filename)

      beta_strand_determination(input_path, output_path)


