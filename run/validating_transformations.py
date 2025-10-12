import y_alignment
import numpy as np
import BestFitLine_Projection
import labels
from sklearn.linear_model import LinearRegression
import sys
y = [1,0]
PIN = sys.argv[1]
Bindices = labels.B_strand_indices()
Bindices =np.array(Bindices)
print(Bindices)
print(f"B shape {Bindices.shape}")
Eindices = labels.E_strand_indices()
Eindices =np.array(Eindices)

X  = y_alignment.y_aligned_protein_BCEF
print(f"X shape:{X.shape}")

transformed_B = BestFitLine_Projection.forming_strand_from_indices(Bindices)
print(transformed_B.shape)
transformed_E = BestFitLine_Projection.forming_strand_from_indices(Eindices)
BE_Strand = np.concatenate((transformed_B,transformed_E),axis=0)
print(BE_Strand.shape)
model = LinearRegression()
model.fit(BE_Strand[:,0].reshape(-1,1),BE_Strand[:,1].reshape(-1,1))
xmin = np.min(BE_Strand[:,0])
xmax = np.max(BE_Strand[:,0])
ymin = model.coef_[0]*xmin+model.intercept_[0]
ymax = model.coef_[0]*xmax+model.intercept_[0]

print(model.coef_)

BE_vector = np.array([xmax,ymax[0]]) - np.array([xmin,ymin[0]])
#BE_vector = np.append(BE_vector)
#print(BE_vector)
def verify_parallel(v1,v2,tol=1e-9):
    v1 = np.array(v1)
    v2 = np.array(v2)
    #print(np.cross(v1, v2))
    return np.isclose(np.cross(v1, v2), 0, atol=tol)
print(verify_parallel(model.coef_[0],y))

#print(y_alignment.angle_with_y_axis(model.coef_))

