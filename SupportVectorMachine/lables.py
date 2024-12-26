import numpy as np 

def Class():
    c = []
    with open('./ATOMlines1CD8_BCEF.txt', 'r') as file:
     for line in file:
            
            if (line[21].strip() == 'B' or line[21].strip() == 'E'):
                c.append(1)
                
            else:
                c.append(-1)
    
    return c


lables = Class()
print(lables)
print(f"Number of Labels: {len(lables)}")
