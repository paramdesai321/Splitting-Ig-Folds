import numpy as np 
import os 
import sys
c = []
B_strand = []
C_strand = []
E_strand = []
F_strand = []
def Label(file_path):
   # file_path = os.path.join(os.path.dirname(__file__),'1CD8_BCEF_ver2.txt')

   # file_path_2 = os.path.join(os.path.dirname(__file__),'ATOMlines2iij_BCEF_backbone_ver2.pdb')
    
    #file_path_3 = os.path.join(os.path.dirname(__file__),'ATOMlines1ifr_BCEF_ver2.pdb')
    #file_path_4 = os.path.join(os.path.dirname(__file__),'ATOMlines1wf5_BCEF_backbone_ver2.pdb')    
    with open(file_path, 'r') as file:
     for i,line in  enumerate(file):
         if(((line[12:16].strip())=="CA")or((line[12:16].strip())=="C")or((line[12:16].strip())=="N")):            
            if (line[21].strip() == 'B' or line[21].strip() == 'E'):
                #print(line[21].strip())
                c.append(1)
            
                if(line[21].strip() == 'B'):
                    B_strand.append(i)                                                   
                if(line[21].strip() == 'E'):
                    E_strand.append(i)

            elif (line[21].strip()=='C' or line[21].strip()=='F'):
                #print(line[21].strip())
                c.append(-1)
                if(line[21].strip() == 'C'):
                    C_strand.append(i)
                if(line[21].strip() == 'F'):
                    F_strand.append(i) 
    
    return c
def get_Labels():
    return c
PIN = sys.argv[1]
def B_strand_indices():
    return np.array(B_strand)
def C_strand_indices():
    return np.array(C_strand)
def E_strand_indices():
    return np.array(E_strand)
def F_strand_indices():
    return np.array(F_strand)
#PIN =  '2iij'
file_path =  os.path.join(os.path.dirname(__file__), f'Beta_Strands/ATOMlines{PIN}_BCEF_Beta.pdb')
lables =Label(file_path)
#print(B_strand_indices())
