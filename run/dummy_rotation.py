import numpy as np
import sklearn_svm as svm
import sys
import parsing_coords 
import CA_C_N_parsing as BCEF_backbone
import labels 
import matplotlib.pyplot as plt
x= np.array([1,-4,2])
coords_from_protein = parsing_coords.coordinates()



#applying rotation 
angle = svm.angle_between_planes(svm.model.coef_[0],[0,0,1])

plane_coords = svm.get_plane_coords()
#print(plane_coords.shape)
#print(svm.apply_rotation_matrix(plane_coords).T)
rotation_matrix = svm.R_z(angle)
#print(rotation_matrix)
reverse_rotation_matrix = svm.R_z(-1*angle)

#print(reverse_rotation_matrix)
X  = BCEF_backbone.coordinates()
y = labels.get_Labels()

## Rotating along y axis
def R_y(angle):         
   angle = 90-angle
   #angle = -1*angle                                                                                              
   c,s = np.cos(angle),np.sin(angle)                                                                                     
   rotation_matrix = np.array([[c,0,-s],                                                                                 
                               [0,1,0],                                                                                  
                                [s,0,c]])                                                                                
   return rotation_matrix                                                   
def R_x(angle):

    c,s = np.cos(angle),np.sin(angle)
    rotation_matrix = np.array([1,0,0],[0,c,-s],[0,s,c])
    return rotation_matrix
                                                                                                                      
def applying_rotation_matrix(axis,grid):                                                                                      
                                                                                                                         
    angle = svm.angle_between_planes(svm.model.coef_[0],[0,1,0]) # protein is not aligned to x axis                 
    return np.dot(axis(angle),np.transpose(grid))                                                                

print("$$$$")
print(plane_coords)
transform = applying_rotation_matrix(R_x,plane_coords)
print(transform.T)
                                                                          
def plot_plane(X,y,model):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax1 = fig.add_subplot(121, projection='3d')
    X = np.array(X)
    # Scatter plot of the points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')

    # Create grid to evaluate model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    print(xlim)
    print("@@@@@@")
    print(ylim)
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 200),
        np.linspace(ylim[0], ylim[1], 200)
    )
    print("SHAPE of xx: {xx.shape}")
    # Calculate corresponding z values assuming the decision boundary is a plane: w1*x + w2*y + w3*z + b = 0
    w = model.coef_[0]
    b = model.intercept_[0]
    params_svm = np.append(w,b)
    # z = (-w1*x - w2*y - b) / w3
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]
    zz = np.array(zz)
    
    zz = svm.apply_rotation_matrix(zz)
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='black')

