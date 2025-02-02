import numpy as np
coords = []
x_coord = []
y_coord = []
z_coord = []


def striping_coords(line, x_coord, y_coord, z_coord):
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    x_coord.append(x)
    y_coord.append(y)
    z_coord.append(z)
    coords.append([x, y, z])


with open('./1CD8_BCEF_ver2.txt', 'r') as rf:
        for line in rf:
            strand = line[21].strip()
             
            if strand == 'B':
                print(line)
                striping_coords(line, x_coord, y_coord, z_coord)


for i in range(len(coords)):
    print(f'Line {i}: {coords[i]}')


def z_coordinates():
    return z_coord

def x_coordinates():
    return x_coord

def y_coordinates():
    return y_coord
print("----------------------")
print(np.array(coords))
coords = np.array(coords)
print(coords.shape)
