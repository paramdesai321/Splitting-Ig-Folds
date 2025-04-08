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

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 30),
        np.linspace(ylim[0], ylim[1], 30)
    )

    # Calculate corresponding z values assuming the decision boundary is a plane: w1*x + w2*y + w3*z + b = 0
    w = model.coef_[0]
    b = model.intercept_[0]

    # z = (-w1*x - w2*y - b) / w3
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]
    # Plotting the Best fit plane for two parts of the beta sheet
    BE,CF = half_plane(X,y)
    w_BE,b_BE = Plane_for_Strands(BE)    
    print(w_BE)    
   # w_BE = params[:3]
   # b_BE = params[3] 
    z_BE = w_BE[0]*xx + w_BE[1]*yy + b_BE
    
    w_CF,b_CF = Plane_for_Strands(CF)
    z_CF = w_CF[0]*xx + w_CF[1]*yy + b_CF
    # Plot the decision boundary plane
    
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='black')
    ax.plot_surface(xx, yy, z_BE, alpha=0.3, color='red')
    ax.plot_surface(xx, yy, z_CF, alpha=0.3, color='blue')
    
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    plt.title(f"SVM Decision Boundary in 3D for PDB {PIN}")
    plt.show()
plot_svm_decision_boundary_3d(X,y,model)
print(rmsd_from_plane(X,model))

