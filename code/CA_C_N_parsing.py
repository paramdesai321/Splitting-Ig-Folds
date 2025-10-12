import os
import sys
z_coord = []
coords = []
def DesiredAtoms(line):
    result = False    
    if(((line[12:16].strip())=="CA")or((line[12:16].strip())=="C")or((line[12:16].strip())=="N")):
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip()) 
            coords.append([x,y,z])           
            print(line)
            result = True
    return result

PIN = sys.argv[1]
#PIN = '2iij' 
file_path = os.path.join(os.path.dirname(__file__),f'./BCEF/{PIN}_BCEF.pdb')
atom_lines = []
#with open(f'./Backbone/ATOMlines{PIN}_BCEF_backbone.pdb','w') as wf:
with open(file_path, 'r') as rf:

        for line in rf:
           atom_lines = [l for l in rf if(DesiredAtoms(line)==True)]
           print(atom_lines)
                                
with open(file_path,'w') as wf:
           wf.writelines(atom_lines)



def x_coordinates():
    return x_coord;

def y_coordinates():
    return y_coord;

def z_coordinates():
    return z_coord
def coordinates():
    return coords
# Testing Suite
print(coords)
#file_input = sys.argv[1]

print(f"Length of the Coords: {len(coords)}")
