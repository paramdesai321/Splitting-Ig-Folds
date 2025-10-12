import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
import CA_C_N_parsing as atoms  
import labels 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import translation_protein
import math
from mpl_toolkits.mplot3d import Axes3D
X = translation_protein.shifted_protein_BCEF   # Use first two features: sepal length and width)
y = labels.get_Labels() 
# Keep only classes 0 and 1 for binary classification
#X = X[y != 2]
#y = y[y != 2]
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train Linear SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)
plane_coords_new = []
## Plotting function)
#def rmsd(X):
#
#       X = np.array(X)                    
#       x2 = (-model.coef_[0] * X[:,0] - model.coef_[0][1] * X[:,1] - model.intercept_[0]) / model.coef_[0][2] # xw0+yw1+zw2 = 0
#       dist_from_plane = (model.coef_[0]*X[:,0] + model.coef_[0][1]*X[:,1] + model.coef_[0][2]*X[:,2])/(model.coef_[0][0]**2 + model.coef_[0][1]**2 + model.coef_[0][2]**2)**1/2
#       
#       print(rmsd)
#       return np.sqrt(np.mean(dist_from_plane**2))
#
PIN = sys.argv[1]
def rmsd_from_plane(X):
    w = model.coef_[0]     # shape (3,)
    b = model.intercept_[0]

    # Numerator: dot product of weights with each point + bias (absolute value)
    numerators = np.abs(np.dot(X, w) + b)

    # Denominator: norm of the normal vector (w)
    denominator = np.linalg.norm(w)

    # Distances from plane
    distances = numerators / denominator

    # RMSD
    rmsd_val = np.sqrt(np.mean(distances ** 2))
    return rmsd_val

#print(model.coef_[0])            
#print(model.coef_[0][0])            
#print(model.coef_[0][1])            
#print(model.coef_[0][2])            
#print(model.intercept_[0])            
def Plane_for_Strands(X):
    
    model = LinearRegression()   
    X = np.array(X) 
    XY = np.column_stack((X[:,0],X[:,1]))
#    print(X)
    
    model.fit(XY,X[:,2])
    return model.coef_,model.intercept_
def half_plane(X,c):
    X_BE = []
    X_CF =[] 
    for i,cl  in enumerate(c):
        if(cl == 1): 
           X_BE.append(X[i])
        else:
           X_CF.append(X[i])
    return X_BE,X_CF
# Angle between Strand and the plane
def angle_between_planes(n1, n2):
   # Avoid divide-by-zero
   # n1 and n2 vectors  are the parmaeters of the planes 
    n1 = np.array(n1)
    n2 = np.array(n2)
    

    # Dot product and norms
    dot_product =np.dot(n1, n2)  # use abs to get angle between 0-90
#    print(f"Dot product in angle : {dot_product}")
    norm_product = np.linalg.norm(n1) * np.linalg.norm(n2)
 #   print(f"norm product in angle : {norm_product}")

    if norm_product == 0:
        raise ValueError("One of the normal vectors is zero-length")

    # Compute angle in radians, then convert to degrees
    cos_theta = dot_product / norm_product
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # clip to avoid domain error
#    angle_deg = np.degrees(angle_rad)

    return angle_rad

# Rotation matrix
def R_z(angle):
    angle =angle
    s,c = np.sin(angle),np.cos(angle)
    matrix = np.array( [
        [c,-s,0],
        [s,c,0],
        [0,0,1]
    ])
    return matrix
    

# Centroid and RMSD
def centroid(X,Y):
    c_x = np.mean(X,axis=0)
    c_y = np.mean(Y,axis=0)
    dist = np.abs(c_x - c_y)
    return np.linalg.norm(dist)
#print(f"X shape: {X.shape}")
def plane_grid(X,y,model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax1 = fig.add_subplot(121, projection='3d')
    X = np.array(X)
    # Scatter plot of the points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    ax.set_box_aspect([1, 1, 1])
    # Create grid to evaluate model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
 #   print(xlim)
 #   print("@@@@@@")
 #   print(ylim)
    xx, yy = np.meshgrid(
        np.linspace(math.floor(np.min(X[:,0])-5), math.ceil(np.max(X[:,0])+5), 200),
        np.linspace(math.floor(np.min(X[:,1])-5),  math.ceil(np.max(X[:,1])+5), 200)
    )
  #  print("SHAPE of xx: {xx.shape}")
    # Calculate corresponding z values assuming the decision boundary is a plane: w1*x + w2*y + w3*z + b = 0

    w = model.coef_[0]
    b = model.intercept_[0]
    params_svm = np.append(w,b)
    # z = (-w1*x - w2*y - b) / w3
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]
    zz = np.array(zz)
    meshgrid = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3) 
    #print(f"zz shape : {zz.shape}")
#    plane_coords.append(xx)
#    plane_coords.append(yy)
#    plane_coords.append(zz)
#    plane_coords_new = np.array(plane_coords)
    #meshgrid = np.stack([xx,yy,zz]) 
   # print(f"mesgrid: {meshgrid}")
    plane_coords =meshgrid
   # print(plane_coords)
    #plane_coords = meshgrid.reshape(3, -1).T
#    print(f"Plane_Coords = {plane_coords}")
#    print(f"Zlim:{zlim}")
    return plane_coords 

def apply_rotation_matrix(grid):
    
    angle = angle_between_planes(model.coef_[0],[0,0,1])
#    angle = 180-angle 
    angle = angle*-1 
#    return np.dot(np.transpose(R_z(angle)),grid)
    return np.dot(R_z(angle),np.transpose(grid)) # taking transpose of the argument


def plot_svm_decision_boundary_3d(X, y,model):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax1 = fig.add_subplot(121, projection='3d')
    X = np.array(X)
    # Scatter plot of the points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')

    # Create grid to evaluate model
    xlim = ax.get_xlim()
  #  print(f"xlim : {xlim}")
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    #print("zlim0")
    #print(zlim)

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 100),
        np.linspace(ylim[0], ylim[1], 100)
    )
    # Calculate corresponding z values assuming the decision boundary is a plane: w1*x + w2*y + w3*z + b = 0
    w = model.coef_[0]
    b = model.intercept_[0]
    params_svm = np.append(w,b)
    # z = (-w1*x - w2*y - b) / w3
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]
    zz = np.array(zz)
    x_min = min(xx[0])
    x_max = max(xx[0])
#    print(f"{zz}")
#    print(f"{x_min}")
#    print(f"{x_max}")
#    print("$$$$$$$")
    y_min = min(yy[0])
    y_max = max(yy[-1])
#    print(f"{y_min}")
#    print(f"{y_max}")
#    print("$$$$$$$")
#    z_min = min(zz[0])
    z_max = max(zz[-1])
#    print(f"{z_min}")
#    print(f"{z_max}")
#    print("$$$$$$$")
##    plane_coords.append(xx)
#    plane_coords.append(yy)
#    plane_coords.append(zz)
#    plane_coords_new = np.array(plane_coords)
    meshgrid = np.stack([xx,yy,zz]) 
    plane_coords_new  =meshgrid
    plane_coords_new = meshgrid.reshape(3, -1).T
    #print(f"Plane_Coords_new = {plane_coords_new}")
    #print(f"Plane_Coords")
    # Plotting the Best fit plane for two parts of the beta sheet
    BE,CF = half_plane(X,y)
    w_BE,b_BE = Plane_for_Strands(BE)
        
   # w_BE = params[:3]
   # b_BE = params[3] 
    z_BE = w_BE[0]*xx + w_BE[1]*yy + b_BE
      
    w_CF,b_CF = Plane_for_Strands(CF)
    z_CF = w_CF[0]*xx + w_CF[1]*yy + b_CF
    #print(f"Params for BE: {w_BE,b_BE}")
    #print(f"Params for CF: {w_CF,b_CF}")
    # Plot the decision boundary plane
    #print("Angle")
    w_BE = np.append(w_BE,1)
    w_CF = np.append(w_CF,1)
    params_CF = np.append(w_CF,b_CF)
    params_BE = np.append(w_BE,b_BE)
    #print(angle_between_planes(params_BE,params_svm)) 
    #print(angle_between_planes(params_CF,params_svm)) 
  #  ax.scatter([0], [0], [0], color='green')
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='black')
    #ax.plot_surface(xx, yy, z_BE, alpha=0.3, color='red')
    #ax.plot_surface(xx, yy, z_CF, alpha=0.3, color='blue')
    # Plotting the xy plane
    z_zeros = np.zeros_like(xx)
#    ax.plot_surface(xx,yy,z_zeros,alpha=0.3,color='red')    
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    plt.title(f"SVM Decision Boundary in 3D for PDB {PIN}")
    plt.show()
#
plot_svm_decision_boundary_3d(X,y,model)

def get_point_on_plane(X, w, b):
    w = w / np.linalg.norm(w)
    x_mean = np.mean(X, axis=0)
    d = (np.dot(w, x_mean) + b)
    projected = x_mean - d * w
    return projected 
# Testing
#def get_plane_coords():
#    plane_coords = make_square_plane_grid() 
#  # center = np.mean(coords,axis=0)
# #  print(f"Center: {center}")
#    plane_coords = np.array(plane_coords)
#
#    return plane_coords

def get_plane_coords():
    plane_coords = plane_grid(X,y,model)
    print(plane_coords.shape)
    return np.array(plane_coords)
def make_square_plane_grid():
    xmin,xmax = (math.floor(np.min(X[:,0])-5),math.ceil(np.max(X[:,0])+5))
    ymin,ymax = (math.floor(np.min(X[:,1])-5),math.ceil(np.max(X[:,1])+5))
    print(f" xmin = {np.min(X[:,0])} and  {xmin}")
    print(f" xmax = {np.max(X[:,0])} and  {xmax}")
    print(f" ymin = {np.min(X[:,1])} and  {ymin}")
    print(f" ymax = {np.max(X[:,1])} and  {ymax}")
    
    coords = []  
    for i in range(xmin,xmax+1):
        for j in range(ymin,ymax+1):
            coords.append([i,j,0.0])
    
    return coords


                        
get_plane_coords()
#print(rmsd_from_plane(X,model))
#BE,CF = half_plane(X,y)
#print(centroid(BE,CF))
#print("----")
#print(plane_coords_new)
#print(len(plane_coords_new))
#grid = plane_grid(X,y,model)
#print(grid.shape)
#print("***********")
#angle = angle_between_planes(model.coef_[0],[0,0,1])
#print(angle_between_planes(model.coef_[0],[0,0,1]))
#print(R_z(angle))
#Matrix  =  np.dot(R_z(angle),np.transpose(grid))
#print(f"Matrix mult: {np.dot(R_z(angle),np.transpose(grid))}") # R_z : dim{3x3} grid: dim{#grid_points x 3} ==> result: dim {3 x 3}
#print(f"{Matrix.shape}")
