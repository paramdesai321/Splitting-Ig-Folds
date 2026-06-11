import sys
import numpy as np
import re
import glob
import os
import empty_dict
x = []
y = []
z = []
coords_dict = {}
atom_line_dict = {}
#input_dir = './small_scale_testing'

#input_dir = './1cd8'
#input_dir = './fixed_v4_AllIgStrands'
input_dir = './CD_HIT_90_Umesh_AllIgStrands'
def ExtractingAtoms(coords,line):
#    x.append(float(line[30:38].strip()))
#    y.append(float(line[38:46].strip()))
#    z.append(float(line[46:54].strip()))

    x = (float(line[30:38].strip()))
    y = (float(line[38:46].strip()))
    z = (float(line[46:54].strip()))
    coords.append([x,y,z])

def line_per_file(PIN):
    lines = []
    infile = f'{input_dir}/{PIN}.pdb'
    with open(infile,'r') as rf:
        for line in rf:
    
         if line.startswith('ATOM'):
              lines.append(line)  
    return lines     
#PIN = sys.argv[1]
atom_lines = []
#with open(f'./Backbone/ATOMlines{PIN}_BCEF_backbone.pdb','w') as wf:
pattern = re.compile(
    r'^(?P<pdb_id>[^_]+)_'       # pdb ID (e.g. 1A22)
    r'(?P<chain>[^_]+)_'         # chain (e.g. B)
    #r'(?P<start>-?\d+)_to_'        # start residue (e.g. 234) use this line if the file has to in the start and end
    r'(?P<start>-?\d+)_'        # start residue (e.g. 234)
    r'(?P<end>\d+(?:[A-Za-z]+)?)\.pdb$'        # end residue (e.g. 324), then “.pdb” at end
)
i =0
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
#    seg    = m.group('seg')
    coords = []
    atom_lines = []
    with open(infile, 'r') as rf: 
        for line in rf: 
            if line.startswith("ATOM"): 
                atom_lines.append(line)
                ExtractingAtoms(coords,line) 
    
 
    #print(f"Extracting Coordinates from {pdb_id}_{chain}_{start}_to_{end}.pdb")
    key = f"{fname}" 
       
    key = os.path.splitext(key)[0]
    coords_dict[key] = coords
    atom_line_dict[key] = atom_lines
    i+=1

def coordinates():
    return coords

coords_dict = empty_dict.prune_dict(coords_dict) # invoking prune_dict method from empty_dict to get rid of empty keys in the dictionary 
atom_line_dict = empty_dict.prune_dict(atom_line_dict) # invoking prune_dict method from empty_dict to get rid of empty keys in the dictionary 
#print(f"Atom Dict")
#print(atom_line_dict['1ADQ_H_117_to_223B'])
def get_atom_lines():
    return atom_lines
# Testing Suite
#print(coords_dict['1ADQ_H_117_to_223B'])
#print(coords)
#print(f"Length of the Coords: {coords.shape}")
#print(len(coords_dict))
#print(coords_dict['1A4K_L_3_107'])

if __name__ == "__main__":
    coords_dict = empty_dict.prune_dict(coords_dict)
    #print(coords_dict.get('1A4K_L_3_107'))
