from pymol import cmd
from pymol.cgo import BEGIN, END, COLOR, ALPHA, NORMAL, VERTEX, TRIANGLE_FAN
import math
def draw_plane(a, b, c, d, size=50.0, name='plane', color = (1.0, 0.0, 0.0), alpha=0.5):
    """
    Draws the plane a*x + b*y + c*z + d = 0 as a square patch of half-edge 'size'.
    """
    # normal vector
    n = [a, b, c]
    norm_n = math.sqrt(a*a + b*b + c*c)
    # point on plane
    p0 = [ -d * a/(norm_n**2), -d * b/(norm_n**2), -d * c/(norm_n**2) ]

    # pick a vector w not parallel to n
    if abs(a) < 0.9:
        w = [1,0,0]
    else:
        w = [0,1,0]
    # u = normalized(n x w)
    u = [ n[1]*w[2] - n[2]*w[1],
          n[2]*w[0] - n[0]*w[2],
          n[0]*w[1] - n[1]*w[0] ]
    norm_u = math.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
    u = [u[i]/norm_u for i in range(3)]

    # v = n x u  (already unit length)
    v = [ n[1]*u[2] - n[2]*u[1],
          n[2]*u[0] - n[0]*u[2],
          n[0]*u[1] - n[1]*u[0] ]

    # build corners
    corners = []
    for dx, dy in [(-size,-size), ( size,-size), ( size, size), (-size, size)]:
        corners.append([
            p0[i] + dx*u[i] + dy*v[i] for i in range(3)
        ])

    # assemble CGO
    obj = [ BEGIN, TRIANGLE_FAN,
            COLOR, color[0], color[1], color[2],
            ALPHA, alpha,
            NORMAL, n[0], n[1], n[2] ]
    for xyz in corners:
        obj += [ VERTEX ] + xyz
    obj.append(END)

    cmd.load_cgo(obj, name)
draw_plane(0,0,1,0)
# make the function available in PyMOL
cmd.extend('draw_plane', draw_plane)

