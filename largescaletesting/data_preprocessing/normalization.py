import sys
sys.path.append('../')
import feature_matrix



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

def index_distance_features(matrix):
   X = matrix[-4:]

def normalize_angles(angle_features):
    result = []
    for feature in angle_features:
        element =  np.cos(feature)
        result.append(feature)
    return np.array(result)



def normalize_distances(dist_features):
    result = []
    for feature in dist_features:
        xmin = np.min(features)
        xmax = np.max(features)
        norm  = (2 * (feature-xmin)/(xmax-xmin)) - 1
        result.append(norm)
    return np.array(result)

X = feature_matrix.X
X = X.T
print(X.shape)
