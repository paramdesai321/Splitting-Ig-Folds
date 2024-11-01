
import parsingcoord as coordinates
B_X= []
B_Y=[]
B_Z=[]
C_X= []
C_Y= []
C_Z = []
E_X = []
E_Y=[]
E_Z = []

F_X = []
F_Y=[]
F_Z=[]


with open('ParsedAtoms.txt','r') as rf:

    for line in rf:
        if(line[21].strip()=='B'):
            coordinates.striping_coords(line,B_X,B_Y,B_Z)
        if(line[21].strip()=='C'):
           coordinates.striping_coords(line,C_X,C_Y,C_Z)
        if(line[21].strip()=='E'):
            coordinates.striping_coords(line,E_X,E_Y,E_Z)
        if(line[21].strip()=='F'):
            coordinates.striping_coords(line,F_X,F_Y,F_Z)



    print(C_X)

