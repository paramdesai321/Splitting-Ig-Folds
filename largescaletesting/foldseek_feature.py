#!/usr/bin/env python3
import numpy as np
ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")  # 20 known 3Di states
STATE_TO_IDX = {s:i for i,s in enumerate(ALPHABET)}
TARGET_CHAINS = ["B", "C", "E", "F"]

def parse_pdb_file(filename):
    pdb_map = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split line into index + PDB ID (some may lack index)
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key_str, pdb_str = parts
            else:
                pdb_str = parts[0]
                key_str = None

            # Clean out binary null characters (\x00) and ^@
            pdb_str = pdb_str.replace("\x00", "").lstrip("^@").strip()

            # Skip if line is empty or only contained nulls
            if not pdb_str:
                continue

            # Only keep valid 4-char PDB-style identifiers
            # Example: 8XFZ_B_3_to_95 → valid, because first 4 are alphanumeric
            if len(pdb_str) < 4 or not pdb_str[:4].isalnum():
                continue

            # Convert index if present
            if key_str and key_str.isdigit():
                pdb_map[int(key_str)] = pdb_str
            else:
                pdb_map[len(pdb_map) + 1] = pdb_str

    return pdb_map



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
def extract_features():
    pdb_map = parse_pdb_file('foldseek_training_set_h')
    feature_dict = {}
    for pdb_index in pdb_map.keys():
        pdb_file = f"aligned_pdbs/final_{pdb_map[pdb_index]}.pdb"  # replace with your filename
        chain_residue = extract_chain_residues(pdb_file)
        foldseek_sequences = map_foldseek_sequences('foldseek_training_set_ss',chain_residue,strand_index=pdb_index)
        B_matrix  = count_vector(foldseek_sequences['B_foldseek_sequence'])
        C_matrix  = count_vector(foldseek_sequences['C_foldseek_sequence'])
        E_matrix  = count_vector(foldseek_sequences['E_foldseek_sequence'])
        F_matrix  = count_vector(foldseek_sequences['F_foldseek_sequence'])
        final_encoding = np.hstack([B_matrix,C_matrix,E_matrix,F_matrix])
        feature_dict[pdb_map[pdb_index]] = final_encoding
    return feature_dict
            
                
feature_dict = extract_features()
print(feature_dict['1B88_A_3_to_114'])
foldseek_matrix = np.array(list(feature_dict.values()))
print(foldseek_matrix.shape)
        

#if __name__ == "__main__":
#    pdb_file = "aligned_pdbs/final_1A4K_L_3_to_107.pdb"  # replace with your filename
#    result = extract_chain_residues(pdb_file)
#    print("\nResult dictionary:\n", result)
#    chain_residues= extract_chain_residues(pdb_file)
#    foldseek_sequences = map_foldseek_sequences('foldseek_training_set_ss', chain_residues)
#    print(one_hot(foldseek_sequences['B_foldseek_sequence'])
#    print(foldseek_sequences)
