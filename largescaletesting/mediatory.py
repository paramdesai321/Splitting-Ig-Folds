import glob
import os
import re
import subprocess
from feature_vector import super_dict

input_dir = './Labels'
pattern = re.compile(
    r'^Labels_'           # literal prefix
    r'(?P<pdb_id>[^_]+)_'     # PDB ID, e.g. 3P40
    r'(?P<chain>[^_]+)_'      # chain, e.g. A
    r'(?P<start>-?\d+)_'       # start residue
    r'(?P<end>\d+(?:[A-Za-z]+)?)_'         # end residue
    r'seg(?P<seg>\d+)\.txt$' # segment number
)

for infile in glob.glob(os.path.join(input_dir,'*.txt')):
    fname = os.path.basename(infile)
    m = pattern.match(fname)
    if not m:
        print(f"Skipping (unrecognized pattern): {fname}")
        continue

    # Extract metadata
    pdb_id = m.group('pdb_id')
    chain  = m.group('chain')
    start  = m.group('start')
    end    = m.group('end')
    seg    = int(m.group('seg'))
    
    label_fname = f"Labels_{pdb_id}_{chain}_{start}_to_{end}_seg{seg}.txt"
    label_path = os.path.join("Labels", label_fname)
    argument = re.sub("Labels_","",label_fname)
    argument = os.path.splitext(argument)[0]
    argument2 = re.sub(r"_seg\d+$","",argument)
 #   print("Arg -")
#    print(argument2)
    
#    print(argument)
    # call sklearn_svm.py on that label file
#    if argument2 in super_dict:
 #       print(f"Skipping {argument2} as it already exists in super_dict")
  #      continue
    subprocess.run(["python3", "create_pdb.py", argument2], check=True)
    subprocess.run(["python3", "plane_pdb.py", argument2], check=True)
    subprocess.run(["python3", "merge_pdb_files.py", argument], check=True)
    subprocess.run(["python3", "main.py", argument2], check=True)
    subprocess.run(["python3", "feature_matrix.py", argument2], check=True)
    subprocess.run(["python3", "features_to_csv.py", argument2], check=True)
