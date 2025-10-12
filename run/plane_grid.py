import numpy as np
import transformation 
import math
import y_alignment
X = y_alignment.y_aligned_protein
X = X.T
print(X.shape)
def make_square_plane_grid():
    xmin,xmax = (math.floor(np.min(X[:,0])-3),math.ceil(np.max(X[:,0])+3))
    ymin,ymax = (math.floor(np.min(X[:,1])-3),math.ceil(np.max(X[:,1])+3))
    print(f" xmin = {np.min(X[:,0])} and  {xmin}")
    print(f" xmax = {np.max(X[:,0])} and  {xmax}")
    print(f" ymin = {np.min(X[:,1])} and  {ymin}")
    print(f" ymax = {np.max(X[:,1])} and  {ymax}")
        
    coords = []  
    for i in range(xmin,xmax+1):
        for j in range(ymin,ymax+1):
            coords.append([i,j,0.0])
         
    return coords
make_square_plane_grid()

