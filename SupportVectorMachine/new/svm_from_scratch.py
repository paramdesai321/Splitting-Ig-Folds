import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../')
from labels import Label
sys.path.append('../../BestFitPlane/src')
#from Strands import B_X,B_Y,B_Z,C_X,C_Y,C_Z,E_X,E_Y,E_Z,F_X,F_Y,F_Z
from CA_C_N_parsing import coordinates
class SVM:
    def __init__(self,learning_rate = 0.01, lambda_param= 0.01, n_iters = 1000):
        self.lr = learning_rate;
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w =None
        self.b = None
        

    def fit(self,X,c): 
        training_length = X.shape
        print(training_length)
        self.w = np.zeros(training_length)
        print(self.w.shape)
        self.w = self.w[1]
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx,x_i in enumerate(X):

                condition = c[idx]* (np.dot(x_i, self.w) - self.b) >= 1
                if condition: 
                    self.w = self.w - self.lr * (2 * self.lambda_param * self.w)
                   # no change in the bias
                else: 
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i,c[idx]))
                    self.b -= self.lr * c[idx]
# what should the input look like according to the fit method
# Z  can be of any length
# Goal: Predicting a plane based of X, Y and Z value
# Previous Mistake: Trying to map Z value based on X and Y values
# Note: Classication isn't regression; y is the class label which is infeered using all the data points; thus you would have to use all three coordinate to predict the class labels 1. B,E 3. C,F
# X = [[x],[y],[z]]
#---------------------
# Rethinking the structure of the data
# X = [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3].............[xn,yn,zn]]

    def predict(self,X,c):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
# the input strucutre sohuld be consistent with predict()
    def plot_hyperplanes(self, X,z): 
        fig = plt.figure(figsize=(10, 6)) 
        ax = fig.add_subplot(111, projection='3d')

        # Plotting the data points
        for idx, label in enumerate(z):
            color = 'b' if label == 1 else 'r' 
            marker = 'o' if label == 1 else 'x' 
            ax.scatter(X[idx][0], X[idx][1], X[idx][2], color=color, marker=marker, s=100)

        # Create a grid for plotting the decision boundary in 3D
        x0_range = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 50) 
        x1_range = np.linspace(np.amin(X[:, 1]), np.amax(X[:, 1]), 50) 
        x0, x1 = np.meshgrid(x0_range, x1_range)
    
        # Calculate the decision boundary (plane equation)
        x2 = (-self.w[0] * x0 - self.w[1] * x1 - self.b) / self.w[2]
    
        # Plot decision boundary (a plane in 3D)
        ax.plot_surface(x0, x1, x2, color='k', alpha=0.5, rstride=100, cstride=100, label="Decision boundary")

        # Axis labels
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.set_title("SVM Decision Boundary in 3D")
        plt.show()


# Testing the SVM with a simple dataset

if __name__ == "__main__":

    # Generate 50 points for Class 1 centered at (5, 5, 5)
#    X =[]
#   X[0] = x_coord
#    X[1] = y_coord
#    X[2] = z_coord
    # Generate 50 points for Class -1 centered at (-5, -5, -5)
#     X_class2 = np.random.randn(50, 3) + [-5, -5, -5]
#     y_class2 = -np.ones(50)
# 
#     # Combine both classes to create the dataset
#     X = np.vstack((X_class1, X_class2))
#     Y = np.vstack((X_class1, X_class2))  # Using the same dataset for both X and Y (you can modify this if needed)
#     z = np.hstack((y_class1, y_class2))
# 
    # Shuffle the dataset to mix class labels
#    indices = np.arange(X.shape[0])
#    np.random.shuffle(indices)
#    X = X[indices]
#    Y = Y[indices]
#    z = z[indices]
#
#    X = x_coord
#    Y = y_coord
#    
#    # Initialize and train the SVM
#    clf = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
#    clf.fit(X, Y, z)
#
#    # Shuffle the dataset to mix class labels
#    indices = np.arange(X.shape[0])
#    np.random.shuffle(indices)
#    X = X[indices]
#    Y = Y[indices]
#    z = z[indices]
#
#    X = x_coord
#    Y = y_coord
#    ct = labels.Label()
#    Input  = CA_C_parsing.coordinates 
#    # Initialize and train the SVM
#    clf = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
#    clf.fit(Input, c)
#
#    # Plot the data points and hyperplanes
#    clf.plot_hyperplanes(Input,c)
     input_data = np.array(coordinates())
     classes = Label()
     clf = SVM(learning_rate=0.01, lambda_param=0.01,n_iters=1000)
     clf.fit(input_data,classes)
     clf.plot_hyperplanes(input_data,classes)
