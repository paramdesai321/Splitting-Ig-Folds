import sys

PIN = sys.argv[1]
path = f'./BCEF/{PIN}_BCEF.pdb'

with open(path, 'r') as rf:
    atom_lines = [l for l in rf if l.startswith('ATOM')]

with open(path, 'w') as wf:
    wf.writelines(atom_lines)

    
