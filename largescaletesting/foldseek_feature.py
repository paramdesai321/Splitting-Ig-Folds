#!/usr/bin/env python3
import numpy as np
from identify_CD19 import cd19_pins
#####CAUTION: PLEASE NOTE THAT Foldseek is run on All Strands not on BCEF for this script!!!!!!!
ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")  # 20 known 3Di states
STATE_TO_IDX = {s:i for i,s in enumerate(ALPHABET)}
TARGET_CHAINS = ["B", "C", "E", "F"]

def parse_ids(filename):
    result = {}
    
    with open(filename, "r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            # Drop index if present (take last token)
            parts = line.split(maxsplit=1)
            value = parts[-1]
            
            # Clean leading junk
            value = value.replace("\x00", "").lstrip("^@").strip()
            
            if value:
                result[i] = value
    
    return result


def one_hot(seq: str):
    vec = [0] * len(ALPHABET)
    for ch in seq:
        if ch in STATE_TO_IDX:
            vec[STATE_TO_IDX[ch]] = 1
    return vec

def extract_chain_residues(pdb_file):
    target_chains = ["B", "C", "E", "F"]
    chain_residues = {ch: [] for ch in target_chains}
    seen = {ch: set() for ch in target_chains}

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain_id = line[21].strip()
                if chain_id in target_chains:
                    resnum_str = line[22:26].strip()
                    if resnum_str.isdigit():
                        resnum = int(resnum_str)
                        # Only add if not already seen for that chain
                        if resnum not in seen[chain_id]:
                            seen[chain_id].add(resnum)
                            chain_residues[chain_id].append(resnum)

    for ch in target_chains:
        print(f"Chain {ch}: {len(chain_residues[ch])} residues")

    return chain_residues

def map_foldseek_sequences(foldseek_file, chain_lines, strand_index=1):
    

    with open(foldseek_file) as f:
        lines = f.readlines()

    # Extract the sequence line for the given strand
    seq_line = lines[strand_index - 1].strip()
    # Remove any numeric prefix or leading symbols (like '1 ', '^@')
    seq = "".join([c for c in seq_line if c.isalpha()])

    foldseek_subseqs = {}
    for ch, indices in chain_lines.items():
        subseq = "".join(seq[i - 1] for i in indices if i - 1 < len(seq))
        foldseek_subseqs[f"{ch}_foldseek_sequence"] = subseq

    return foldseek_subseqs
def count_vector(seq: str):
    counts = [0] * len(ALPHABET)
    for ch in seq:
        if ch in STATE_TO_IDX:
            counts[STATE_TO_IDX[ch]] += 1
    return counts

def is_cd19(pdb_name):
    return pdb_name in set(cd19_pins)


def extract_features():
    pdb_map = parse_ids('foldseek_CD_HIT_90_AllStrands_h_randomly_ordered')
    #pdb_map = parse_ids('foldseek_CD_HIT_90_AllStrands_h_reordered')
    #pdb_map = parse_ids('foldseek_CD_HIT_90_AllStrands_h')
    cd19_set = set(cd19_pins)
    feature_dict = {}
    for pdb_index in pdb_map.keys():
        pdb_name = pdb_map[pdb_index]
        if pdb_name in cd19_set:
            continue
        pdb_file = f"aligned_pdbs_CD_HIT_90/aligned_{pdb_map[pdb_index]}_seg0.pdb"  # replace with your filename
        chain_residue = extract_chain_residues(pdb_file)
        foldseek_sequences = map_foldseek_sequences('foldseek_CD_HIT_90_AllStrands_ss',chain_residue,strand_index=pdb_index)
        B_matrix  = count_vector(foldseek_sequences['B_foldseek_sequence'])
        C_matrix  = count_vector(foldseek_sequences['C_foldseek_sequence'])
        E_matrix  = count_vector(foldseek_sequences['E_foldseek_sequence'])
        F_matrix  = count_vector(foldseek_sequences['F_foldseek_sequence'])
        final_encoding = np.hstack([B_matrix,C_matrix,E_matrix,F_matrix])
        feature_dict[pdb_map[pdb_index]] = final_encoding
    cd19_set = set(cd19_pins)
    foldseek_set = set(pdb_map.values())

    print("Total Foldseek entries:", len(foldseek_set))
    print("Total CD19 pins:", len(cd19_set))

    matched_cd19 = cd19_set & foldseek_set
    missing_cd19 = cd19_set - foldseek_set

    print("CD19 pins found in Foldseek header:", len(matched_cd19))
    print("CD19 pins missing from Foldseek header:", len(missing_cd19))

    if missing_cd19:
        print("\nMissing CD19 pins:")
        for pin in sorted(missing_cd19):
            print(repr(pin))
    return feature_dict
            
                
feature_dict = extract_features()
#print(feature_dict['1B88_A_3_to_114'])
foldseek_matrix = np.array(list(feature_dict.values()))
print(foldseek_matrix.shape)
from identify_CD19 import cd19_pins, not_cd19

cd19_set = set(cd19_pins)
not_cd19_set = set(not_cd19)
#pdb_map = parse_ids('foldseek_CD_HIT_90_AllStrands_h_reordered')
pdb_map = parse_ids('foldseek_CD_HIT_90_AllStrands_h_randomly_ordered')
#pdb_map = parse_ids('foldseek_CD_HIT_90_AllStrands_h')
foldseek_non_cd19 = set()

for pdb_index in pdb_map.keys():
    pdb_name = pdb_map[pdb_index]

    # skip CD19
    if pdb_name in cd19_set:
        continue

    foldseek_non_cd19.add(pdb_name)

# --- comparison ---
extra_in_foldseek = foldseek_non_cd19 - not_cd19_set
missing_from_foldseek = not_cd19_set - foldseek_non_cd19

print(f"\nFoldseek non-CD19 count: {len(foldseek_non_cd19)}")
print(f"Expected not_cd19 count: {len(not_cd19_set)}")

print(f"\nExtra in Foldseek (should NOT be there): {len(extra_in_foldseek)}")
for x in sorted(extra_in_foldseek):
    print(x)

print(f"\nMissing from Foldseek (expected but not found): {len(missing_from_foldseek)}")
for x in sorted(missing_from_foldseek):
    print(x)        

#if __name__ == "__main__":
#    pdb_file = "aligned_pdbs/final_1A4K_L_3_to_107.pdb"  # replace with your filename
#    result = extract_chain_residues(pdb_file)
#    print("\nResult dictionary:\n", result)
#    chain_residues= extract_chain_residues(pdb_file)
#    foldseek_sequences = map_foldseek_sequences('foldseek_training_set_ss', chain_residues)
#    print(one_hot(foldseek_sequences['B_foldseek_sequence'])
#    print(foldseek_sequences)
