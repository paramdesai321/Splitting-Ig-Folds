import sklearn as svm
import labels
import numpy as np
import CA_C_N_parsing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transformation
import translation_plane
import translation_protein
import strand_projection
from sklearn.linear_model import LinearRegression
Bindices = labels.B_strand_indices()
Cindices = labels.C_strand_indices()
Eindices = labels.E_strand_indices()
Findices = labels.F_strand_indices()
X = transformation.transformed_protein_BCEF
X = np.array(X)


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
    print(f"X shape : {X.shape}")    
    # 1. Select the coordinates for only the atoms belonging to the B-strand.
    b_strand_coords = X[Bindices, :]
    
    # 2. Select the Z-coordinates (the 3rd column, index 2) of those atoms.
    z_coords = b_strand_coords[:, 2]
    
    # 3. Calculate the average of those Z-coordinates.
    average = np.mean(z_coords)

    # --- Original lines commented out by Gemini ---
    # average=0
    # z= X[2] 
    # average = np.sum(X[:len(Bindices)])
    # average = average/len(Bindices)
    
    if average>=0:
        return True
    else:
        return False
def flipping_transformation(protein):
    if(is_sign_of_B_strand_positive()==True):
         return protein
    else:
        return np.dot(protein,R_y(np.pi))
flipped_protein_BCEF = flipping_transformation(X)
X = flipped_protein_BCEF

def forming_strand_from_indices(indices):
    strand = []
    for index in indices:
        strand.append(X[index])
    strand = np.array(strand)
    return strand

B_Strand=  forming_strand_from_indices(Bindices)
E_Strand=  forming_strand_from_indices(Eindices)
BE_Strand = np.concatenate((B_Strand,E_Strand),axis=0)
print(BE_Strand.shape)
#print(f"BE  : {BE_Strand}")i
model = LinearRegression()
model.fit(BE_Strand[:,0].reshape(-1,1),BE_Strand[:,1].reshape(-1,1))
xmin = np.min(BE_Strand[:,0])
xmax = np.max(BE_Strand[:,0])
ymin = model.coef_[0]*xmin+model.intercept_[0]
ymax = model.coef_[0]*xmax+model.intercept_[0]

BE_vector = np.array([xmax,ymax[0]]) - np.array([xmin,ymin[0]])
BE_vector = np.append(BE_vector,0)

#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='blue', label='Tranformed Protein BCEF')
#
#ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color='red', label='Best-Fit Line')
#
#ax.scatter(*centroid, color='green', s=100, label='Centroid')
#
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#ax.set_title('Best-Fit Line through CA Coordinates')
#
#ax.legend()
#plt.tight_layout()
#plt.show()
#line_projection = []
#or i in range(len(line_points)):
   #np.append(line_projection,strand_projection.project_point_to_plane_implicit(line_points[i],strand_alignment.get_shifted_plane()[:3],strand_alignment.get_shifted_plane()[3:]))
#    line_projection.append(strand_projection.project_point_to_plane_implicit(line_points[i],np.array([0,0,1]),0))
#rint(np.array(line_projection))
def get_best_fit_line_projection():
   np.linespace(-5,5)
    
   return np.array(line_projection)

def equation_of_line(p1,p2):
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]
    return np.array([m,b]) # returning m and b packed 

def get_params_of_line_projection():
#   global line_projection
#   line_projection = np.array(line_projection)
    params = np.append(model.coef_,model.intercept_) 
    params = np.append(params,0)
    return params
def strand_vector(indicies):                          
    X = flipped_protein_BCEF                
    X = np.array(X)                                   
    first,last = indicies[0],indicies[-1]               
    vector = X[last] - X[first]                         
    return vector                                       
def dot_product(direction,B_strand_vector):             
    u1 = direction/np.linalg.norm(direction)            
    u2 = B_strand_vector/np.linalg.norm(B_strand_vector)
    return np.dot(u1,u2)                               
B_Strand_Vector = strand_vector(Bindices)
line = get_params_of_line_projection()
print(f"Line  : {line}")
print(f"BE Vector: {BE_vector}")
print(f"B: {strand_vector(Bindices)}")
#print(f"C: {strand_vector(Cindices)}")
#print(f"E: {strand_vector(Eindices)}")
#print(f"F: {strand_vector(Findices)}")
