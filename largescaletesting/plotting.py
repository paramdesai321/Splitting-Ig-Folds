from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import non_colinear
import translation_plane
import numpy as np
from matplotlib import cm
import labels
def plot_plane(arg, xlim=(-10, 10), ylim=(-10, 10), resolution=100, color='red', ax=None,label='Plane'):
    if isinstance(arg, (list, tuple,np.ndarray)) and len(arg) == 4 and all(isinstance(x, (int, float)) for x in arg):
        a, b, c, d = arg
    else:
        points = non_colinear.find_non_collinear_points_fast(arg)
        a, b, c, d = translation_plane.plane_equation_from_points(points[0], points[1], points[2])

    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    x, y = np.meshgrid(x, y)
    z = (d - a*x - b*y) / c

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, alpha=0.7, color=color, edgecolor='k',label=label)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Plane {a}x + {b}y + {c}z = {d}')

    if ax is None:
        plt.show()
    return ax



def plot_points_3d(points,y=None, color='red', size=50, ax=None,label='Points'):
    
    points = np.array(points)
    X = points
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Must be a list of 3D points (x, y, z)")


    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    if y== None:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=size, color=color,label=label)
    else:    
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, s=size,label=label)
    ax.legend()
    return ax




            
       
