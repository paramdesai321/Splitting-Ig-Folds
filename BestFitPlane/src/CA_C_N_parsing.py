import parsingcoord
x_coord=[]
y_coord=[]
z_coord = []
def DesiredAtoms(line):

        if(((line[12:16].strip())=="CA")or((line[12:16].strip())=="C")or((line[12:16].strip())=="N")):
            parsingcoord.striping_coords(line,x_coord,y_coord,z_coord)

            print(line)
            return True

with open(f'ParsedAtoms.txt','w') as wf:

    with open(f'../1CD8_BCEF_ver2.pdb', 'r') as rf:

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
