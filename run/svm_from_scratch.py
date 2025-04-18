import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import labels
#from Strands import B_X,B_Y,B_Z,C_X,C_Y,C_Z,E_X,E_Y,E_Z,F_X,F_Y,F_Z
from CA_C_N_parsing import coordinates


    
class SVM:
    
    
    def __init__(self,learning_rate = 0.01, lambda_param= 0.01, n_iters = 1000):
        self.lr = learning_rate;
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w =None
        self.b = None
        self.w_init = None
        self.b_init = None
        
    def plane_initialization(self,X):
        X = np.array(X)
        X = X[:4]
        
        X = np.column_stack((X, np.ones(X.shape[0])))
        #print(X)
        y = np.zeros(X.shape[0])
        #print(y)
        weights = np.linalg.solve(X,y)
        return weights 
            
    def fit(self,X,c): 
        n_samples, n_features= X.shape
        #print(n_samples)
        self.w = np.zeros(n_features)
        init_weights  = self.plane_initialization(X)
        self.w = init_weights[:3]
        self.b = init_weights[3:]
        self.w_init = init_weights[:3]
        self.b_init = init_weights[3:]

        #print(self.w.shape)
        #print(self.w)
        
        cmin=0.0
        cmax=0.0 
        for _ in range(self.n_iters):
            for idx,x_i in enumerate(X):
                #print(f'x_i {idx} = {x_i}')
                #print(f'{idx} conditon value =  {c[idx]* (np.dot(x_i, self.w) - self.b)}')
<<<<<<< HEAD
                print(f"weights: {self.w}")
                condition_val = (c[idx] *( np.dot(x_i, self.w) + self.b))
                #print(condition_val)
=======
               # print(self.w)
                #print(c[idx])
                condition_val = (c[idx] * np.dot(x_i, self.w) + self.b)
>>>>>>> sklearn
                if idx == 0:
                   cmin=condition_val
                   cmax=condition_val
                else:
                   if condition_val < cmin:
                      cmin = condition_val
                   if condition_val > cmax:
                      cmax = condition_val
                condition =  (c[idx] * np.dot(x_i, self.w) + self.b) >=  1
               # print(f'{idx} condition vale = {condition}')
                if condition: 
                    self.w -=  self.lr * (2 * self.lambda_param * self.w)
                   # no change in the bias
                # c[idx]* abs(np.dot(x_i, self.w) - self.b) <=1
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i,c[idx]))
                    self.b -= self.lr * c[idx]
        print(cmin,cmax)   
        print(f"Self.w =  {self.w}")
        print(f"self.w[0] = {self.w[0]}")
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
    def class_metric(self,prediction):
        count_pos = 0
        count_neg = 0
        for elem in prediction:
            if elem > 0:
               count_pos+=1
            else:
               count_neg+=1
        return count_pos,count_neg
    def correct_predictions(self,prediction,lables):
         pos_samples,neg_samples = self.class_metric(prediction)
         print(f"# of Prediction for class 1: {pos_samples}")
         
         print(f"# of Prediction for class 2: {neg_samples}")
         print("----------------------------------------------")
         pos_labels,neg_labels = self.class_metric(classes)
         print(f"# of actual samples  for class 1: {pos_labels}")
         print(f"# of actual samples  for class 2: {neg_labels}")
         print("----------------------------------------------")
         print("Loss in class 1:")
         print(abs(pos_samples-pos_labels))
         print("----------------------------------------------")
         print("Loss in class 2:")
         print(abs(neg_samples-neg_labels))       
# the input strucutre sohuld be consistent with predict()
    
    def rmsd(self,X):

        X = np.array(X)                
        x2 = (-self.w[0] * X[:,0] - self.w[1] * X[:,1] - self.b) / self.w[2] # xw0+yw1+zw2 = 0
        dist_from_plane = (self.w[0]*X[:,0] + self.w[1]*X[:,1] + self.w[2]*X[:,2])/(self.w[0]**2 + self.w[1]**2 + self.w[2]**2)**1/2
        return np.sqrt(np.mean(dist_from_plane**2))
        
    
        print(rmsd)
    def half_plane(self,X,c):
        X_BE = []
        X_CF =[]
        for i,cl  in enumerate(c):
            if(cl == 1):
                X_BE.append(X[i])
            else:
                X_CF.append(X[i])
        return X_BE,X_CF
    def plane_correction(self,X,c,step_size=0.1,max_iters=10000):
        min_loss = float('inf')
        best_b = self.b

        for _ in range(max_iters):
         X_BE, X_CF = self.half_plane(X, c)
         rmsd_BE = self.rmsd(X_BE)
         rmsd_CF = self.rmsd(X_CF)
         loss = (rmsd_BE - rmsd_CF) ** 2  # Smooth loss function

         if loss < min_loss:
            min_loss = loss
            best_b = self.b

<<<<<<< HEAD
        b = best_b
        print(f"The Value of b after gradient free search: {b}")
=======
        # Compute gradient using finite differences
         epsilon = 1e-5
         self.b += epsilon
         X_BE_eps, X_CF_eps = self.half_plane(X, c)
         rmsd_BE_eps = self.rmsd(X_BE_eps)
         rmsd_CF_eps = self.rmsd(X_CF_eps)
         loss_eps = (rmsd_BE_eps - rmsd_CF_eps) ** 2
         self.b -= epsilon

         gradient = (loss_eps - loss) / epsilon

        # Update b using gradient descent
         self.b -= step_size * gradient

        self.b = best_b
>>>>>>> sklearn
        print(f"Optimized b: {self.b}, Final Loss: {min_loss}")
        return self.b    
    def plot_hyperplanes(self, X, z):
     fig = plt.figure(figsize=(10, 6))
     ax1 = fig.add_subplot(131, projection='3d')
     ax2 = fig.add_subplot(132, projection='3d')
     ax3 = fig.add_subplot(133, projection='3d')

     for idx, label in enumerate(z):
        color = 'b' if label == 1 else 'r'
        marker = 'o' if label == 1 else 'x'
<<<<<<< HEAD
        ax1.scatter(X[idx][0], X[idx][1], X[idx][2], color=color, marker=marker, s=100)
        ax2.scatter(X[idx][0], X[idx][1], X[idx][2], color=color, marker=marker, s=100)
        ax3.scatter(X[idx][0], X[idx][1], X[idx][2], color=color, marker=marker, s=100)

     x0 = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 50)
     x1 = np.linspace(np.amin(X[:, 1]), np.amax(X[:, 1]), 50)
     #x0 = X[:,0]
     #print(f"shape of x0 : {x0.shape}")
     #x1 =  X[:,1]
     #print(f"shape of x1: {x1.shape}")
     x0, x1 = np.meshgrid(x0,x1)
    # print(f"x0_range: {x0_range}")
    # print(x0)
    # print(X[:,0])
     x2_old = (self.w[0] * x0 - self.w[1] * x1  - self.b) / self.w[2]
     print(f"SELF.B = {self.b}")
     print(f"SVM: {x2_old}")
     print(f"shape of {x2_old.shape}")

     ax2.plot_surface(x0, x1, x2_old, color='b', alpha=0.5, rstride=100, cstride=100)
=======
        ax.scatter(X[idx][0], X[idx][1], X[idx][2], color=color, marker=marker, s=100)
     x0_range = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 50)
     x1_range = np.linspace(np.amin(X[:, 1]), np.amax(X[:, 1]), 50)
     x0, x1 = np.meshgrid(x0_range, x1_range)
     print(f'x0 : {x0}')
     print(f'x1 : {x1}')
     print(len(x1))

     x2_old = (-self.w[0] * x0 - self.w[1] * x1 - self.b) / self.w[2]
#    ax.plot_surface(x0, x1, x2_old, color='b', alpha=0.5, rstride=100, cstride=100)
     #ax.plot_surface(x0, x1,np.random.rand(50,50), color='b')
     print(self.w)
>>>>>>> sklearn

     b_new = self.plane_correction(X, z)
     x2_new = (-self.w[0] * x0 - self.w[1] * x1 - b_new) / self.w[2]
     print(f"Gradient Free Search: {x2_new}")
     ax3.plot_surface(x0, x1, x2_new, color='r', alpha=0.5, rstride=100, cstride=100)

<<<<<<< HEAD
     #x2_init = (0*x0 - 0*x1 -0) / self.w_init[2]
     
     x2_init = (-self.w_init[0]*x0 - self.w_init[1]*x1 - self.b_init) / self.w_init[2]
     print(f"Init: {x2_init}")
     ax1.plot_surface(x0,x1,x2_init, color = 'y', alpha = 0.5, rstride=100, cstride=100)   
     ax1.set_xlabel("x")
     ax1.set_ylabel("y")
     ax1.set_zlabel("z")
     ax1.set_title("Init Plane in 3D")
     ax2.set_title("SVM Decision Boundaries in 3D")
     ax3.set_title("SVM Decision Boundaries with Gradient-Free Search in 3D")


     # For Legends 
     legend_elements = [
                
        plt.Line2D([0], [0], color='y', lw=2, label='init Plane'),

        plt.Line2D([0], [0], color='b', lw=2, label='Original Plane'),
        plt.Line2D([0], [0], color='r', lw=2, label='Corrected Plane')
=======
     ax.set_xlabel("x")
     ax.set_ylabel("y")
     ax.set_zlabel("z")
     ax.set_title("SVM Plane for Protein 1cd8")
#    
     legend_elements = [
        plt.Line2D([0], [0], color='b', lw=2, label='B,E Strands'),
        plt.Line2D([0], [0], color='r', lw=2, label='C,F Strands')
>>>>>>> sklearn
    ]
     ax1.legend(handles=legend_elements)
     ax2.legend(handles=legend_elements)
     ax2.legend(handles=legend_elements)
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
     print("###################")
    # print(input_data)
     PIN = sys.argv[1]
     file_input_for_labels = os.path.join(os.path.dirname(__file__), f'Beta_Strands/ATOMlines{PIN}_BCEF_Beta.pdb')
     classes = labels.Label(file_input_for_labels)
<<<<<<< HEAD
     clf = SVM(learning_rate=0.01, lambda_param=0.01,n_iters=10)
=======
     clf = SVM(learning_rate=0.001, lambda_param=0.1,n_iters=30)
>>>>>>> sklearn
     clf.fit(input_data,classes)
     prediction = clf.predict(input_data,classes)
     print(prediction)
     clf.correct_predictions(prediction,classes) 
     clf.half_plane(input_data,classes)
     clf.plane_correction(input_data,classes)
     print("RMSD")
     print(clf.rmsd(input_data))
     clf.plot_hyperplanes(input_data,classes)   
    
