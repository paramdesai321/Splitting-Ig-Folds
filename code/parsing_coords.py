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
file_path = f"./PDB/{PIN}.pdb"

with open(file_path, 'r') as rf:
    for line in rf:
        if line.startswith('ATOM'):
            ExtractingAtoms(line)

#coords = np.stack([x, y, z])

def coordinates():
    return coords
coords = np.array(coords)

# Testing Suite
print(coords)
print(f"Length of the Coords: {coords.shape}")

