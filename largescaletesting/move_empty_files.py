import re
import os
import sys
import subprocess
import glob
desired_files  = []

input_dir = './emptyATOMlines'
target_dir = './train_test_IgAllStrands_Umesh'                                                                                                                                            
output_dir = './empty_original_pdbs'
pattern = re.compile(
    r'^ATOMlines_'               # literal prefix
    r'(?P<pdb_id>[^_]+)_'        # pdb ID (e.g. 3P40)
    r'(?P<chain>[^_]+)_'         # chain (e.g. A)
    r'(?P<start>-?\d+)_to_'           # start residue (e.g. 39)
    r'(?P<end>\d+(?:[A-Za-z]+)?)_'             # end residue (e.g. 137)
    r'seg(?P<seg>\d+)\.pdb$'      # segment number (e.g. seg0.pdb)
)
for infile in glob.glob(os.path.join(input_dir, '*.pdb')):                                                                                                                                     
    fname = os.path.basename(infile)                                                                                                                                                 
    m = pattern.match(fname)                                                                                                                                                              
    if not m:                                                                                                                                                                              
        print(f"Skipping (unrecognized pattern): {fname}")                                                                                                                                 
        continue                                                                                                       
                                                                                                                                                                                          
    pdb_id = m.group('pdb_id')                                                                                                                                                             
    chain  = m.group('chain')                                                                                                                                                              
    start  = m.group('start')                                                                                                                                                              
    end    = m.group('end')                                                                                                                                                                
    #seg    = m.group('seg')
    fname =  f"{pdb_id}_{chain}_{start}_to_{end}.pdb"

    subprocess.run(["mv",f"{target_dir}/{fname}",output_dir],check=True)


