import CA_C_N_parsing
import sklearn_svm as svm
import transformation
#import BestFitLine_Projection
import labels
import translation_plane
import translation_protein
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotting
import BestFitLine_Projection
import sys
PIN = sys.argv[1]
def R_z(theta):
   
    c, s = np.cos(theta), np.sin(theta)
    # both np functions take the angle is rads 
    R = np.array([
        [ c,-s ,0],
        [ s, c,  0],
        [0, 0,  1]
    ])
    return R
def R_y(theta):
    c,s = np.cos(theta),np.sin(theta)
    R = np.array([                                                                                                                 
        [ c,0 ,s],                                                                                                                  
        [ 0, 1,  0],                                                                                                                
        [-s, 0,  c]                                                                                                                 
    ])  
    return R
def rotation(protein,rotation_matrix,tol=1e-7):
    result = np.matmul(protein,rotation_matrix)
    result[np.abs(result) < tol] = 0.0
    return result
 
def dot_product(direction,B_strand_vector):
    u1 = direction/np.linalg.norm(direction)
    u2 = B_strand_vector/np.linalg.norm(B_strand_vector)
    return np.dot(u1,u2)
def angle_with_y_axis(line_eqn):
#    i2b = dot_product(line_eqn, BestFitLine_Projection.B_Strand_Vector)
#    if i2b < 0:
#       line_eqn = -line_eqn
#    m = line_eqn[0]   # Don't want use line but vectors
    y_vector = np.array([0, 1])
    line_eqn = np.array(line_eqn)

    dot = np.dot(line_eqn, y_vector)
    norm = np.linalg.norm(line_eqn) * np.linalg.norm(y_vector) 
    # The angle to rotate by is the negative of the angle the vector makes with the y-axis.
    # For a vector (x,y), the angle with the y-axis is arctan2(x,y).
    # We rotate by the negative of that angle to align it with the y-axis.
    #theta = np.arctan2(line_eqn[0],1) # Correct
    
#   # rotation_angle = np.pi/2 - theta # Debatable
#    theta = np.arccos(dot/norm)
#    if np.sign(line_eqn) == (np.array([1,1]) or np.array([1,-1])):
#        return theta
#    else if np.sign(line_eqn) == (np.array([-1,1]) or np.array([-1,-1])):
#        return (2*np.pi) - theta
#
    theta = np.arctan2(line_eqn[0], line_eqn[1])  # (-π, π]
    return theta      
    #print(f"Final Angle: {np.degrees(rotation_angle)}")
    #print(f"Final Angle: {np.degrees(theta)}")
#angle = angle_with_y_axis(BestFitLine_Projection.model.coef_[0])
#angle = angle_with_y_axis(BestFitLine_Projection.BE_vector)
angle = angle_with_y_axis(BestFitLine_Projection.B_vector)
#print(f"Transformed Protein BCEF shape : {transformation.transformed_protein_BCEF.shape}")
print(f"angle from y axis: {np.degrees(angle)}")
#y_aligned_protein_BCEF = np.dot(R_z(angle),transformation.transformed_protein_BCEF.T)
y_aligned_protein_BCEF = rotation(transformation.transformed_protein_BCEF,R_z(-angle))
#BE_vector_y_aligned = rotation(np.append(BestFitLine_Projection.BE_vector,0),R_z(-angle))
#print("BE",BE_vector_y_aligned)
print(BestFitLine_Projection.B_vector)
B_vector_y_aligned = rotation(np.append(BestFitLine_Projection.B_vector,0),R_z(-angle))

print(B_vector_y_aligned)
with open('Not_Y_aligned.md', 'a') as f:
    if(B_vector_y_aligned[0] != 0.0 or B_vector_y_aligned[2] !=0.0):
        f.write(f'{PIN}\n')


print(y_aligned_protein_BCEF.shape)
#print(angle_with_y_axis(BE_vector_y_aligned))
#print("DDDDDDDDD")
#print(len(y_aligned_protein_BCEF[0]))
#print("-------------------")
#print("Transformed Protein shape: ",transformation.transformed_protein.shape)
y_aligned_protein = rotation(transformation.transformed_protein,R_z(-angle))
#y_aligned_protein =  np.dot(R_z(angle),transformation.transformed_protein.T)
y_aligned_protein = np.array(y_aligned_protein)
#print("@@@@@@@@@@@@@@@@@@")
#print(y_aligned_protein.shape)
#print(labels.B_strand_indices())
#X = CA_C_N_parsing.coordinates()
#
#
#print("Actual Plane")
#print(f"First 5:{X[:5]}")
#print(f"Last: 5 : {X[-5:]}")
#print("Translated Plane")
#print(f"First {translation_protein.shifted_protein_BCEF[:5]}")
#print(f"Last {translation_protein.shifted_protein_BCEF[-5:]}")
#print(f"Tranformation Plane")
#print(f"Transformed aligned to xy: first  {transformation.transformed_protein_BCEF[:5]}")
#print(f"Transformed aligned to xy: last  {transformation.transformed_protein_BCEF[-5:]}")
#print(f"Transformed aligned to xy: first  {y_aligned_protein_BCEF.T[:5]}")
#print(f"Transformed aligned to xy: first  {y_aligned_protein_BCEF.T[-5:]}")
#print("-------")
#print(len(transformation.transformed_protein_BCEF))
#
#flipped_protein = flipping_transformation(transformation.transformed_protein_BCEF)    
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#points = transformation.transformed_protein_BCEF
#plotting.plot_points_3d(points, ax=ax)
#plotting.plot_points_3d(X, ax=ax,label='X')
#
#plotting.plot_points_3d(translation_plane.shifted_protein_BCEF_cm, ax=ax,label='shifted BCEF',color='black')
#plotting.plot_points_3d(transformation.transformed_protein_BCEF,ax=ax,label='Tranformed BCEF',color='black')
#plotting.plot_points_3d(y_aligned_protein.T, ax=ax,label='Y Aligned Protein',color='blue')
#plotting.plot_points_3d(y_aligned_protein_BCEF.T,y=labels.get_Labels(f"{PIN}_seg0"), ax=ax,label='Y Aligned BCEF',color='red')
#plotting.plot_points_3d(svm.make_square_plane_grid(),ax=ax,label='Y Aligned BCEF',color='red')
#plotting.plot_points_3d(BestFitLine_Projection.get_best_fit_line_projection(), ax=ax,label='Line Projection',color='red')
#plotting.plot_points_3d(svm.get_plane_coords(), ax=ax,label='Actual',color='black')
#grid =  np.linspace(-1, 1, 100)
#param = BestFitLine_Projection.BE_vector
#y  = param[0]*grid + param[1]
#plt.plot(grid, y, label='best fit line') 
#
#ax.scatter([0], [0], [0], color='green')
#plotting.plot_plane(np.concatenate((svm.model.coef_[0],svm.model.intercept_)),ax=ax,label='actual plane')
#plotting.plot_plane(translation_plane.shifted_plane,ax=ax,label='shifted plane')
#plotting.plot_plane(transformation.transformed_plane,ax=ax,label='Tranformed plane')
#plt.show()
#
#        
#direction_best_fit_line  = BestFitLine_Projection.direction
#theta = angle_with_y_axis(direction_best_fit_line)
#dt = dot_product(direction_best_fit_line,B_strand_vector(labels.B_strand_indices()))
#print(dt)
#print(theta)
#aligned_plane = np.dot(strand_alignment.shifted_plane,R_y(theta))
#
#aligned_protein = np.dot(strand_alignment.shifted_protein,R_y(theta))
#print(labels.B_strand_indices())
#print(aligned_protein)
#print(f"aligned : {y_aligned_protein_BCEF}")
#
