import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../BestFitPlane/src')
import CA_C_N_parsing as atoms  
sys.path.append('../../SupportVectorMachine')
import labels 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Load Iris dataset and use only 2 classes and 2 features for easy plotting
iris = datasets.load_iris()
X = atoms.coordinates()   # Use first two features: sepal length and width
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
def rmsd_from_plane(X, model):
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
#
def Plane_for_Strands(X):
    model = LinearRegression()   
    X = np.array(X) 
    XY = np.column_stack((X[:,0],X[:,1]))
    print(XY)
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
    dot_product = np.abs(np.dot(n1, n2))  # use abs to get angle between 0-90
    norm_product = np.linalg.norm(n1) * np.linalg.norm(n2)

    if norm_product == 0:
        raise ValueError("One of the normal vectors is zero-length")

    # Compute angle in radians, then convert to degrees
    cos_theta = dot_product / norm_product
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # clip to avoid domain error
    angle_deg = np.degrees(angle_rad)

    return angle_deg
# Centroid and RMSD
def centroid(X,Y):
    c_x = np.mean(X,axis=0)
    c_y = np.mean(Y,axis=0)
    dist = np.abs(c_x - c_y)
    return np.linalg.norm(dist)
 
def plane_grid(X,y,model):
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
        np.linspace(xlim[0], xlim[1], 20),
        np.linspace(ylim[0], ylim[1], 20)
    )
    print("SHAPE of xx: {xx.shape}")
    # Calculate corresponding z values assuming the decision boundary is a plane: w1*x + w2*y + w3*z + b = 0
    w = model.coef_[0]
    b = model.intercept_[0]
    params_svm = np.append(w,b)
    # z = (-w1*x - w2*y - b) / w3
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]
    zz = np.array(zz)
#    plane_coords.append(xx)
#    plane_coords.append(yy)
#    plane_coords.append(zz)
#    plane_coords_new = np.array(plane_coords)
    meshgrid = np.stack([xx,yy,zz]) 
    print(f"mesgrid: {meshgrid}")
    plane_coords =meshgrid
    plane_coords = meshgrid.reshape(3, -1).T
    print(f"Plane_Coords = {plane_coords}")
    print(f"Zlim:{zlim}")
    return plane_coords 
def plot_svm_decision_boundary_3d(X, y, model):
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
    print("zlim0")
    print(zlim)

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 30),
        np.linspace(ylim[0], ylim[1], 30)
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
    print(f"{zz}")
    print(f"{x_min}")
    print(f"{x_max}")
    print("$$$$$$$")
    y_min = min(yy[0])
    y_max = max(yy[-1])
    print(f"{y_min}")
    print(f"{y_max}")
    print("$$$$$$$")
    z_min = min(zz[0])
    z_max = max(zz[-1])
    print(f"{z_min}")
    print(f"{z_max}")
    print("$$$$$$$")
#    plane_coords.append(xx)
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
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='black')
    #ax.plot_surface(xx, yy, z_BE, alpha=0.3, color='red')
    #ax.plot_surface(xx, yy, z_CF, alpha=0.3, color='blue')
        
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    plt.title(f"SVM Decision Boundary in 3D for PDB {PIN}")
    plt.show()
    print(f"Zlim:{zlim}")

plot_svm_decision_boundary_3d(X,y,model)
    
# Testing
def get_plane_coords():
   plane_coords = plane_grid(X,y,model)
   return plane_coords
print(rmsd_from_plane(X,model))
BE,CF = half_plane(X,y)
print(centroid(BE,CF))
print("----")
print(plane_coords_new)
print(len(plane_coords_new))
plane_grid(X,y,model)


