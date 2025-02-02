import numpy as np
import sys
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from lables import Class
sys.path.append('../BestFitPlane/src')
#from Strands import B_X,B_Y,B_Z,C_X,C_Y,C_Z,E_X,E_Y,E_Z,F_X,F_Y,F_Z
from CA_C_N_parsing import x_coord,y_coord,z_coord

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w1 = None  # Weights for the first feature set (X)
        self.w2 = None  # Weights for the second feature set (Y)
        self.b = None   # Bias term

    def fit(self, X,c):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx,x_i in enumerate(Z):
                condition = c[idx] * (np.dot(x_i, self.w)- self.b) >= 1
                if condition:
                    # Regularization term for w1 and w2
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Update rule when the margin is violated
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, c[idx]))
                    self.b -= self.lr * c[idx]

    def predict(self, X, c):
        approx = np.dot(X, self.w)- self.b
        return np.sign(approx)

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
        x2 = (-self.w1[0] * x0 - self.w1[1] * x1 - self.b) / self.w1[2]
        
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
    X =[]
    X[0] = x_coord
    X[1] = y_coord
    X[2] = z_coord
    # Generate 50 points for Class -1 centered at (-5, -5, -5)
    X_class2 = np.random.randn(50, 3) + [-5, -5, -5]
    y_class2 = -np.ones(50)

    # Combine both classes to create the dataset
    X = np.vstack((X_class1, X_class2))
    Y = np.vstack((X_class1, X_class2))  # Using the same dataset for both X and Y (you can modify this if needed)
    z = np.hstack((y_class1, y_class2))

    # Shuffle the dataset to mix class labels
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    z = z[indices]
    
    X = x_coord
    Y = y_coord
    
    # Initialize and train the SVM
    clf = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    clf.fit(X, Y, z)
    
    print(X)
    # Plot the data points and hy:perplanes
    clf.plot_hyperplanes(X, Y, z)
  
