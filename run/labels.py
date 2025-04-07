import numpy as np 
import os 
import sys
c = []
def Label(file_path):
   # file_path = os.path.join(os.path.dirname(__file__),'1CD8_BCEF_ver2.txt')

   # file_path_2 = os.path.join(os.path.dirname(__file__),'ATOMlines2iij_BCEF_backbone_ver2.pdb')
    
    #file_path_3 = os.path.join(os.path.dirname(__file__),'ATOMlines1ifr_BCEF_ver2.pdb')
    #file_path_4 = os.path.join(os.path.dirname(__file__),'ATOMlines1wf5_BCEF_backbone_ver2.pdb')    
    with open(file_path, 'r') as file:
     for i,line in  enumerate(file):
         if(((line[12:16].strip())=="CA")or((line[12:16].strip())=="C")or((line[12:16].strip())=="N")):            
            if (line[21].strip() == 'B' or line[21].strip() == 'E'):
                print(line[21].strip())
                c.append(1)
                
            elif (line[21].strip()=='C' or line[21].strip()=='F'):
                print(line[21].strip())
                c.append(-1)
    
    return c
def get_Labels():
    return c
#PIN = sys.argv[1]
PIN =  '1cd8'
file_path =  os.path.join(os.path.dirname(__file__), f'Beta_Strands/ATOMlines{PIN}_BCEF_Beta.pdb')
lables =Label(file_path)
print(lables)
print(f"Number of Labels: {len(lables)}")
