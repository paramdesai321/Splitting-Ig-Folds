import os
import sys
z_coord = []
coords = []
def DesiredAtoms(line):

        if(((line[12:16].strip())=="CA")or((line[12:16].strip())=="C")or((line[12:16].strip())=="N")):
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip()) 
            coords.append([x,y,z])           
            print(line)
            return True
file_path_1 = os.path.join(os.path.dirname(__file__), '../1CD8_BCEF_ver2.pdb')   
file_path_2 = os.path.join(os.path.dirname(__file__), 'ATOMlines2iij_BCEF.txt')  

file_path_3 = os.path.join(os.path.dirname(__file__), '../1t6v_BCEF.pdb')  
file_path_4 = os.path.join(os.path.dirname(__file__), '../ATOMlines1ifr.txt')

file_path_5 = os.path.join(os.path.dirname(__file__), '../ATOMlines1wf5_BCEF.txt')
PIN = sys.argv[1]
#PIN = '2iij' 
file_path = os.path.join(os.path.dirname(__file__),f'./ATOMlines/ATOMlines{PIN}.pdb')
with open(f'./Backbone/ATOMlines{PIN}_BCEF_backbone.pdb','w') as wf:

    with open(file_path, 'r') as rf:

        for line in rf:

            #DesiredAtoms(line)


            #print(line)
            if(DesiredAtoms(line)==True):
                print("---------------------------")
                wf.write(line)





def x_coordinates():
    return x_coord;

def y_coordinates():
    return y_coord;

def z_coordinates():
    return z_coord
def coordinates():
    return coords
print(coords)
#file_input = sys.argv[1]

print(f"Length of the Coords: {len(coords)}")
