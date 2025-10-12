import sys
import numpy as np

x = []
y = []
z = []
coords = []
def ExtractingAtoms(line):
#    x.append(float(line[30:38].strip()))
#    y.append(float(line[38:46].strip()))
#    z.append(float(line[46:54].strip()))

    x = (float(line[30:38].strip()))
    y = (float(line[38:46].strip()))
    z = (float(line[46:54].strip()))
    coords.append([x,y,z])
PIN = sys.argv[1]
file_path = f"{PIN}.pdb"
atom_lines = []
with open(file_path, 'r') as rf:
    for line in rf:
        if line.startswith('ATOM'):
            atom_lines.append(line)
            ExtractingAtoms(line)

#coords = np.stack([x, y, z])

def coordinates():
#    coords = np.array(coords)
    return np.array(coords)
coords = np.array(coords)
#print(coords)
def get_atom_lines():
    return atom_lines
#print(f"Length of the Coords: {coords.shape}")

