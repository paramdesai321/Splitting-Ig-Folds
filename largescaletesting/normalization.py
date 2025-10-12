import sys

#sys.path.append('../')
import feature_matrix
import numpy as np


def index_angle_between_strand_features(matrix):
    X = matrix[:5]
    return X

def index_dist_cm__features(matrix):
    X = matrix[-4:]
    return X

def index_strand_projection_features(matrix):
    X = matrix[5:8]
    return X

def index_strand_angles_with_x_features(matrix):
    X1 = matrix[8]
    X2 = matrix[11]
    X3 = matrix[14]
    X4 = matrix[17]
    return X1,X2,X3


def index_angle_features(matrix):
    X = matrix[:20]
    return X
def index_distance_features(matrix):
   X = matrix[-4:]
   return X
def normalize_angles(angle_features):
    result = []
    for feature in angle_features:
        element =  np.cos(feature)
        result.append(element)
     
    return np.array(result)



def normalize_distances(dist_features):
    result = []
    for feature in dist_features:
        xmin = np.min(feature)
        xmax = np.max(feature)
        norm  = (2 * (feature-xmin)/(xmax-xmin)) - 1
        result.append(norm)
    return np.array(result)

X = feature_matrix.export_X()
print(f"X shape:{X.shape}")
X= X
print(index_angle_features(X).shape)
normalized_angles = normalize_angles(index_angle_features(X))
print(normalized_angles.shape)
print(np.max(normalized_angles))
#print(index_distance_features(X).shape)
normalized_distances = normalize_distances(index_distance_features(X))
#print(normalized_distances.shape)

normalized_features = np.vstack((normalized_angles,normalized_distances))
print(np.min(normalized_features))
print(np.max(normalized_features))
print(normalized_features.shape)

degree_angles = index_angle_features(X)*(180/np.pi)
degree_distances = index_distance_features(X)*(180/np.pi)
degree_features= np.vstack((degree_angles,degree_distances))




