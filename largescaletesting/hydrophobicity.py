from Bio.PDB import PDBParser
import numpy as np
import argparse
import sys
import feature_vector
import labels
PIN = sys.argv[1]
CLASSES = [
    "NonPolar",
    "Ala",
    "Pro",
    "Gly",
    "Aromatic",
    "Polar",
    "Negative",
    "Positive",
    "Cys"
]

CLASS_MAP = {
    "ILE": "NonPolar",
    "LEU": "NonPolar",
    "VAL": "NonPolar",
    "MET": "NonPolar",
    "ALA": "Ala",
    "PRO": "Pro",
    "GLY": "Gly",
    "CYS": "Cys",
    "PHE": "Aromatic",
    "TRP": "Aromatic",
    "TYR": "Aromatic",
    "SER": "Polar",
    "THR": "Polar",
    "ASN": "Polar",
    "GLN": "Polar",
    "ASP": "Negative",
    "GLU": "Negative",
    "ARG": "Positive",
    "LYS": "Positive",
    "HIS": "Positive",
}


def extract_residues(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    # Initialize fixed chain groups
    residues_by_chain = {"B": [], "C": [], "E": [], "F": []}

    for model in structure:
        for chain in model:
            chain_id = chain.id.strip()
            if chain_id not in residues_by_chain:
                continue  # ignore other chains (A, H, etc.)

            for res in chain:
                hetflag, resseq, icode = res.id
                if hetflag.strip():  # skip HETATM, water, etc.
                    continue

                resname = res.get_resname().upper()
                if resname in CLASS_MAP:
                    residues_by_chain[chain_id].append(resname)

    return residues_by_chain
def one_hot_encode(residues):
    num_classes = len(CLASSES)
    onehot = np.zeros((num_classes, num_classes), dtype=int)

    # check membership for each fixed class
    for i, cls in enumerate(CLASSES):
        found = any(CLASS_MAP.get(res, None) == cls for res in residues)
        if found:
            onehot[i, i] = 1  # mark that this class is present

    return onehot

pdb_file = f"final_transformation_CD_HIT_90/{PIN}_seg0_withPlane.pdb"
def one_hot_matrix(strand:str,pdb_file=pdb_file):
    residue = extract_residues(pdb_file)
    onehot_matrix = one_hot_encode(residue[strand])
    return np.array(onehot_matrix)

def export_hydrophobicity(strand:str,matrix):
    for i,group in enumerate(CLASSES):
        feature_vector.extract_feature(PIN,f'Hydrophobicity {strand}: {CLASSES[i]}',matrix[i][i])
 
B_one_hot_matrix = one_hot_matrix('B')
C_one_hot_matrix = one_hot_matrix('C')
E_one_hot_matrix = one_hot_matrix('E')
F_one_hot_matrix = one_hot_matrix('F')
export_hydrophobicity('B',B_one_hot_matrix)
export_hydrophobicity('C',C_one_hot_matrix)
export_hydrophobicity('E',E_one_hot_matrix)
export_hydrophobicity('F',F_one_hot_matrix)
print(B_one_hot_matrix)
#print(residue["B"])
#print(residue["C"])
#print(residue["E"])
#print(residue["F"])
#print(residue)
#print(onehot_matrix.shape)
#print(onehot_matrix)
#classes = set(CLASS_MAP.values())
#classes = np.array(list(classes))
#print(classes)
#print(len(onehot_matrix[1]))
