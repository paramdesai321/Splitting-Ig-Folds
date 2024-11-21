

with open(f'../BestFitPlane/1CD8_BCEF_ver2.pdb', 'r') as rf:
    with open(f'ATOMlines1CD8_BCEF.txt', 'w') as wf:

     for line in rf:
        if line.startswith('ATOM'):
            #print(line)
            wf.write(line)
