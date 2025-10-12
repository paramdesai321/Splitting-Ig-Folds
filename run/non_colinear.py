import numpy as np
#import strand_alignment 
#from  transformation import get_shifted_plane
def are_collinear(p1, p2, p3, tol=1e-8):
    v1 = p2 - p1
    v2 = p3 - p1
    return np.linalg.norm(np.cross(v1, v2)) < tol

def find_non_collinear_points_fast(points, max_attempts=1000):
    m = points.shape[0]
    if m < 3:
        raise ValueError("Need at least 3 points.")

    idx = np.random.choice(m, size=min(10, m), replace=False)
    base_points = points[idx]

    for i in range(len(base_points)):
        for j in range(i + 1, len(base_points)):
            p1, p2 = base_points[i], base_points[j]
            # Try a third point randomly
            for _ in range(max_attempts):
                k = np.random.randint(0, m)
                p3 = points[k]
                if not are_collinear(p1, p2, p3):
                    return np.array([p1, p2, p3])
    
    raise ValueError("Failed to find non-collinear triplet after many attempts.")

#points_for_plane_equation  = find_non_collinear_points_fast(strand_alignment.get_shifted_plane)
#print(points_for_plane_equation)
