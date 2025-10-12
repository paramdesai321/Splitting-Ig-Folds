import sys

#
#if len(sys.argv) < 2:
#    print("Please Enter a Protien Identification Number to get the ATOM Lines")
#    sys.exit(1)
#
#
PIN = sys.argv[1]
atom_lines = []
with open(f'./dir_BCEF/{PIN}_BCEF.pdb', 'r') as rf:
    with open(f'./ATOMlines/ATOMlines{PIN}.pdb', 'w') as wf:

     for line in rf:
        if line.startswith('ATOM'):
            #print(line)
            atom_lines.append(line)
            wf.write(line)

def get_atom_lines():
    return atom_lines

#print(len(atom_lines))
