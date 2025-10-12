import os
import glob
import re

#input_dir  = '../small_scale_testing_BCEF'


#input_dir = './1cd8_BCEF'

input_dir = '../train_test_IgBCEF_Jiyao'
output_dir = './ATOMlines'
os.makedirs(output_dir, exist_ok=True)

def res_to_chain_map(PIN):
    map_dict = {}
    res_id_start_idx  = 22      # columns 23–26 in 1-based indexing
    res_id_end_idx    = 33      # columns 34–37 in 1-based indexing
    chain_id_idx      = 21      # column 22 in 1-based indexing

    infile = f"{input_dir}/{PIN}_BCEF.pdb"
    with open(infile,'r') as rf:
        for line in rf:
            if not line.startswith("SHEET"):
                continue
            res_id_start = int(line[res_id_start_idx:res_id_start_idx+4].strip())
            res_id_end   = int(line[res_id_end_idx:res_id_end_idx+4].strip())
            chain_id     = line[chain_id_idx].strip()
            iters = res_id_end - res_id_start
            temp = res_id_start
            
            for i in range(iters+1):   # +1 to include the endpoint
                key = temp
                map_dict[key] = chain_id
                temp += 1
    #print(map_dict)
#    print(f"Mapping dict: {map_dict}")
    return map_dict

# regex to parse filenames like 2ZKW_A_2_to_152_BCEF_seg0.pdb
pattern = re.compile(
    r'^(?P<pdb_id>[^_]+)_'        # pdb ID
    r'(?P<chain>[^_]+)_'          # chain
    r'(?P<start>-?\d+)_to_'         # start residue
    r'(?P<end>-?\d+(?:[A-Za-z]+)?)_BCEF.pdb'              # end residue
)
#print(pattern)
for infile in glob.glob(os.path.join(input_dir, '*.pdb')):
    fname = os.path.basename(infile)
    m = pattern.match(fname)
    if not m:
        print(f"Skipping (unrecognized pattern): {fname}")
        continue

    pdb_id = m.group('pdb_id')
    chain  = m.group('chain')
    start  = m.group('start')
    end    = m.group('end')
#    seg    = m.group('seg')

#    print(f"→ Processing PDB {pdb_id}, chain {chain}, residues {start}–{end}")

    outfile = os.path.join(
        output_dir,
        f'ATOMlines_{pdb_id}_{chain}_{start}_to_{end}_seg0.pdb'
        #f'ATOMlines_{pdb_id}_{chain}_{start}_to_{end}_seg{seg}.pdb'
    )

    with open(infile, 'r') as rf, open(outfile, 'w') as wf:
        for line in rf:
            if line.startswith('ATOM'):
                wf.write(line)

    #print(f"   Wrote ATOM lines to {outfile}")
#print(res_to_chain_map('1A4K_L_3_to_107'))
