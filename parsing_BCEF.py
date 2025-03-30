import sys

#
#if len(sys.argv) < 2:
#    print("Please Enter a Protien Identification Number to get the ATOM Lines")
#    sys.exit(1)
#
#
#PIN = sys.argv[1]
#
PIN = input("Enter Protein Identification Number: ")

with open(f'{PIN}_BCEF.pdb', 'r') as rf:
    with open(f'ATOMlines{PIN}.txt', 'w') as wf:

     for line in rf:
        if line.startswith('ATOM'):
            #print(line)
            wf.write(line)
