import numpy as np 
import os 
def Label():
    c = []
    file_path = os.path.join(os.path.dirname(__file__),'1CD8_BCEF_ver2.txt')

    file_path_2 = os.path.join(os.path.dirname(__file__),'ATOMlines2iij_BCEF_backbone_ver2.pdb')
    
    file_path_3 = os.path.join(os.path.dirname(__file__),'ATOMlines1ifr_BCEF_ver2.pdb')
    file_path_4 = os.path.join(os.path.dirname(__file__),'ATOMlines1wf5_BCEF_backbone_ver2.pdb')
    
    with open(file_path_4, 'r') as file:
     for line in file:
         if(((line[12:16].strip())=="CA")or((line[12:16].strip())=="C")or((line[12:16].strip())=="N")):            
            if (line[21].strip() == 'B' or line[21].strip() == 'E'):
                print(line[21].strip())
                c.append(1)
                
            elif (line[21].strip()=='C' or line[21].strip()=='F'):
                print(line[21].strip())
                c.append(-1)
    
    return c


lables =Label()
print(lables)
print(f"Number of Labels: {len(lables)}")
