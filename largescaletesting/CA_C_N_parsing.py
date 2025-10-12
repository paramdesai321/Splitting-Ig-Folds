import glob
import os
import sys
import re 
import empty_dict


input_dir = './ATOMlines'
output_dir = './Backbone'
coords_dict = {}




def DesiredAtoms(coords,line):
    result = False    
    if(((line[12:16].strip())=="CA")or((line[12:16].strip())=="C")or((line[12:16].strip())=="N")):
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip()) 
            coords.append([x,y,z])           
#            print(line)
            result = True
    return result

#PIN = sys.argv[1]
#PIN = '2iij' 
#file_path = os.path.join(os.path.dirname(__file__),f'./BCEF/{PIN}_BCEF.pdb')
atom_lines = []
#with open(f'./Backbone/ATOMlines{PIN}_BCEF_backbone.pdb','w') as wf:
pattern = re.compile(
    r'^ATOMlines_'               # literal prefix
    r'(?P<pdb_id>[^_]+)_'        # pdb ID (e.g. 3P40)
    r'(?P<chain>[^_]+)_'         # chain (e.g. A)
    r'(?P<start>-?\d+)_to_'           # start residue (e.g. 39)
    r'(?P<end>\d+(?:[A-Za-z]+)?)_'             # end residue (e.g. 137)
    r'seg(?P<seg>\d+)\.pdb$'      # segment number (e.g. seg0.pdb)
)
i = 0
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
    seg    = m.group('seg')


    outfile = os.path.join(
        output_dir,
        f'Backbone_{pdb_id}_{chain}_{start}_to_{end}_seg{seg}.pdb'
    )
    coords = []
    with open(infile, 'r') as rf, open(outfile, 'w') as wf:
        for line in rf:
            if DesiredAtoms(coords,line)==True:
                wf.write(line)

#    print(f"   Wrote Backbone to {outfile}")
#with open(file_path, 'r') as rf:
#
#        for line in rf:
 #          atom_lines = [l for l in rf if(DesiredAtoms(line)==True)]
#           print(atom_lines)
                               
#with open(file_path,'w') as wf:
#           wf.writelines(atom_lines)
    key = re.sub(r'^ATOMlines_', '', fname)
    
    key = os.path.splitext(key)[0]
#    print(key)
    coords_dict[key] = coords
#print(f"Processed {len(coords_dict)} files; coordinates stored in coords_dict.")
def x_coordinates():
    return x_coord;
def y_coordinates():
    return y_coord;

def z_coordinates():
    return z_coord
def coordinates():
    return coords

coords_dict = empty_dict.prune_dict(coords_dict)
# Testing Suite
#print(coords.shape)
#file_input = sys.argv[1]
#print(coords_dict)

