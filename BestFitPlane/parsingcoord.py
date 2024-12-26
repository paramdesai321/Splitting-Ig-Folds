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


with open('./1CD8_BCEF_ver2.pdb', 'r') as rf:
        for line in rf:
            strand = line[21].strip()
            strand1 = line[20].strip()
            print(strand1)
            print(strand)
            print(line) 
            if strand == 'A':
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
def coords():
    return coords
print(coords)
print(len(coords))
