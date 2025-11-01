import sklearn as svm
import labels
import numpy as np
import CA_C_N_parsing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transformation
import translation_plane
import translation_protein
from sklearn.svm import SVC
#import strand_projection
from sklearn.linear_model import LinearRegression
import sys
PIN = sys.argv[1]
Bindices = labels.B_strand_dict[f"{PIN}_seg0"]
Cindices = labels.C_strand_dict[f"{PIN}_seg0"]
Eindices = labels.E_strand_dict[f"{PIN}_seg0"]
Findices = labels.F_strand_dict[f"{PIN}_seg0"]
#print(f"Bindices : {Bindices}")
X = transformation.transformed_protein_BCEF # (n,3)
X = np.array(X)
#print(f"Shape of the input X(transformed_BCEF) : {X.shape}") 
#

def R_y(theta):
   c,s = np.cos(theta),np.sin(theta)
   R = np.array([                                                                                                                              
       [ c,0 ,s],                                                                                                                             
       [ 0, 1,  0],                                                                                                                            
       [-s, 0,  c]                                                                                                                             
   ])   
   return R

def is_sign_of_B_strand_positive():
   # --- GEMINI FIX: Correcting the logic to get the average Z-coordinate of the B-strand ---
#    print(f"X shape : {X.shape}")    
   # 1. Select the coordinates for only the atoms belonging to the B-strand.
   b_strand_coords = X[Bindices, :]
   #print(f"B strand coords : {b_strand_coords}") # Correct
   
   # 2. Select the Z-coordinates (the 3rd column, index 2) of those atoms.
   z_coords = b_strand_coords[:, 2] # Correcr
   #print(f"Z coords",z_coords)
   # 3. Calculate the average of those Z-coordinates.
   average = np.mean(z_coords) # Correct (since the hyperplane is z=0 it does not matter what statistic we use) 


   # --- Original lines commented out by Gemini ---
   # average=0
   # z= X[2] 
   # average = np.sum(X[:len(Bindices)])
   # average = average/len(Bindices)
   
   if average>=0:
       return True
   else:
       return False
#print(is_sign_of_B_strand_positive())
def flipping_transformation(protein):
   if(is_sign_of_B_strand_positive()==True):
        return protein
   else:
       return np.matmul(protein,R_y(np.pi)) # Previously dot, dot and matmul gives the same ouput in this case
flipped_protein_BCEF = flipping_transformation(X)
X = flipped_protein_BCEF
X = np.array(X)
#print(f"Flipped BCEF shape: {X.shape}")

def forming_strand_from_indices(X, indices):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] != 3:
        X = X.T
    indices = np.array(indices)
    return X[indices, :]

B_Strand=  forming_strand_from_indices(X,Bindices)

#print(f"Shape of B strand: {B_Strand.shape}")
E_Strand=  forming_strand_from_indices(X,Eindices)
#print(f"Shape of E strand: {E_Strand.shape}")
BE_Strand = np.concatenate((B_Strand,E_Strand),axis=0)
B_labels = np.ones(B_Strand.shape[0], dtype=int)
E_labels = -np.ones(E_Strand.shape[0], dtype=int)
#imin = np.argmin(B_Strand[:,0]) # min and max index based on x
#imax = np.argmax(B_Strand[:,0])
#B_vector = B_Strand[:,:2][imin] - B_Strand[:,:2][imax]
BE_labels = np.concatenate((B_labels, E_labels), axis=0)
#print(f"Shape of BE Strand : {BE_Strand[:,0].reshape(-1,1).shape}")
#print(BE_Strand.shape)
#print(f"BE  : {BE_Strand}")i
#model = LinearRegression()
model  = SVC(kernel='linear') 
#model.fit(BE_Strand[:,0].reshape(-1,1),BE_Strand[:,1].reshape(-1,1))
model.fit(BE_Strand[:,:2],BE_labels)

xmin = np.min(BE_Strand[:,0]) # Check for Ambiguity in this 
xmax = np.max(BE_Strand[:,0])  # Check for Ambiguity in this 
ymin = model.coef_[0]*xmin+model.intercept_[0] # Check for Ambiguity in this 
ymax = model.coef_[0]*xmax+model.intercept_[0] # Check for Ambiguity in this 

BE_vector = np.array([xmax,ymax[0]]) - np.array([xmin,ymin[0]])
#print(f"BE Vector best fit line:",BE_vector)
#BE_vector = np.append(BE_vector,0)
def orientation_of_B_vector(B_vector):
    y_comp = B_vector[1]
    if y_comp >=0:
        return B_vector
    else:
        return -B_vector

B_vector = B_Strand[-1,:2] - B_Strand[0,:2]
#B_vector_oriented = orientation_of_B_vector(B_vector)
#with open('B_vector_change.md','a') as f:
#    if not np.array_equal(B_vector, B_vector_oriented):
#        f.write(f'{PIN}\n')    
#print("Before:",B_vector)
#B_vector = orientation_of_B_vector(B_Strand[-1,:2] - B_Strand[0,:2])
#print(f"After:",B_vector)
##
##fig = plt.figure()
##ax = fig.add_subplot(111, projection='3d')
##
##ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='blue', label='Tranformed Protein BCEF')
##
##ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color='red', label='Best-Fit Line')
##
##ax.scatter(*centroid, color='green', s=100, label='Centroid')
##
##ax.set_xlabel('X')
##ax.set_ylabel('Y')
##ax.set_zlabel('Z')
##ax.set_title('Best-Fit Line through CA Coordinates')
##
##ax.legend()
##plt.tight_layout()
##plt.show()
##line_projection = []
##or i in range(len(line_points)):
#   #np.append(line_projection,strand_projection.project_point_to_plane_implicit(line_points[i],strand_alignment.get_shifted_plane()[:3],strand_alignment.get_shifted_plane()[3:]))
##    line_projection.append(strand_projection.project_point_to_plane_implicit(line_points[i],np.array([0,0,1]),0))
##rint(np.array(line_projection))
#def get_best_fit_line_projection():
#    xmin = np.min(BE_Strand[:,0])
#    xmax = np.max(BE_Strand[:,0])
#    x_vals = np.linspace(xmin, xmax, 100)
#    y_vals = model.coef_[0] * x_vals + model.intercept_[0]
#    z_vals = np.zeros_like(x_vals)
#    line_projection = np.vstack((x_vals, y_vals.flatten(), z_vals)).T
#    return line_projection
#
#def equation_of_line(p1,p2):
#    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
#    b = p1[1] - m * p1[0]
#    return np.array([m,b]) # returning m and b packed 
#
#def get_params_of_line_projection():
##   global line_projection
##   line_projection = np.array(line_projection)
#    params = np.append(model.coef_,model.intercept_) 
#    params = np.append(params,0)
#    return params
#def strand_vector(indicies):                          
#    X = flipped_protein_BCEF                
#    X = np.array(X)                                   
#    first,last = indicies[0],indicies[-1]               
#    vector = X[last] - X[first]                         
#    return vector                                       
#def dot_product(direction,B_strand_vector):             
#    u1 = direction/np.linalg.norm(direction)            
#    u2 = B_strand_vector/np.linalg.norm(B_strand_vector)
#    return np.dot(u1,u2)                               
#B_Strand_Vector = strand_vector(Bindices)
#line = get_params_of_line_projection()
##print(f"Line  : {line}")
##print(f"BE Vector: {BE_vector}")
##print(f"B: {strand_vector(Bindices)}")
##print(f"C: {strand_vector(Cindices)}")
##print(f"E: {strand_vector(Eindices)}")
##print(f"F: {strand_vector(Findices)}")
