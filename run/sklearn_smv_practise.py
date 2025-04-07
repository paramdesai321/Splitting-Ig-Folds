import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
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

# Plotting function


def plot_svm_decision_boundary_3d(X, y, model):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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

    # Plot the decision boundary plane
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='black')

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    plt.title("SVM Decision Boundary in 3D")
    plt.show()
plot_svm_decision_boundary_3d(X,y,model)
