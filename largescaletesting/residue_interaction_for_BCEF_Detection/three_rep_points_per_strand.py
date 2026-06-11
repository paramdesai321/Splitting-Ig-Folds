import numpy as np
from pathlib import Path
import pprint
from Bio.PDB import PDBParser
import centroid_for_each_strand
import nearest_c_alpha
import detect_strands_to_dict
import os
import sys

BACKBONE_ATOMS = ("N", "CA", "C")
pdb_path = sys.argv[1]
def backbone_centroid_for_residue(chain, resseq: int):
    """
    Compute centroid of N, CA, C for residue with given residue number.
    Returns np.array([x,y,z]) or None if residue/atoms are missing.
    """
    res_obj = None
    for r in chain:
        if r.id[0] != " ":
            continue
        if r.id[1] == resseq:
            res_obj = r
            break

    if res_obj is None:
        return None

    coords = []
    for aname in BACKBONE_ATOMS:
        if res_obj.has_id(aname):
            coords.append(res_obj[aname].get_coord())

    if not coords:
        return None

    coords = np.array(coords, dtype=float)
    return coords.mean(axis=0)
def get_nearest_c_alpha_residue_per_strand(centroids, c_alpha_per_strand, strands):
    """
    centroids: {strand_idx: np.array([x,y,z])}
    c_alpha_per_strand: {strand_idx: [[x,y,z], ...]}  (same order as strands[strand_idx])
    strands: {strand_idx: [resnum1, resnum2, ...]}

    Returns:
      nearest[strand_idx] = {
         "resnum": <int>,          # representative residue number
         "coord": [x,y,z],         # its CA coordinate
         "distance": <float>
      }
    """
    nearest = {}

    for k in centroids:
        centroid = np.asarray(centroids[k], dtype=float)
        ca_coords = np.asarray(c_alpha_per_strand[k], dtype=float)  # (n,3)

        dists = np.linalg.norm(ca_coords - centroid, axis=1)
        i = int(np.argmin(dists))

        nearest[k] = {
            "resnum": int(strands[k][i]),
            "coord": ca_coords[i].tolist(),
            "distance": float(dists[i]),
        }

    return nearest

def get_nearest_c_alpha_to_ref_c_alpha(s_ref, s, centroids, c_alpha_per_strand):
    """
    Find nearest C-alpha in strand s to the centroid C* of strand s_ref.

    Returns:
        (index_in_s, coord_in_s)
    """
    ref_centroid = np.asarray(centroids[s_ref], dtype=float)
    tgt_coords = np.asarray(c_alpha_per_strand[s], dtype=float)

    dists = np.linalg.norm(tgt_coords - ref_centroid, axis=1)
    i = int(np.argmin(dists))

    return i, tgt_coords[i].tolist()


def get_triplet_backbone_centroids_per_strand(pdb_path, chain_id, centroids, c_alpha_per_strand, strands):
    """
    For each strand k:
      - find representative residue = nearest CA to strand centroid (within strand)
      - compute:
          c1 = backbone centroid at resnum-1
          c2 = backbone centroid at resnum
          c3 = backbone centroid at resnum+1
    Returns:
      out[k] = {
        "rep_resnum": resnum,
        "c1": [x,y,z] or None,
        "c2": [x,y,z] or None,
        "c3": [x,y,z] or None,
      }
    """
    # Parse structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    # First, get representative residue per strand
    nearest = get_nearest_c_alpha_residue_per_strand(centroids, c_alpha_per_strand, strands)

    out = {}
    for k, rep in nearest.items():
        r = rep["resnum"]

        c1 = backbone_centroid_for_residue(chain, r - 1)
        c2 = backbone_centroid_for_residue(chain, r)
        c3 = backbone_centroid_for_residue(chain, r + 1)

        out[k] = {
            "rep_resnum": r,
            "c1": None if c1 is None else c1.tolist(),
            "c2": None if c2 is None else c2.tolist(),
            "c3": None if c3 is None else c3.tolist(),
        }

    return out

def get_triplets_relative_to_ref_strand(pdb_path, chain_id, s_ref, centroids, c_alpha_per_strand, strands):
    """
    Build triplets for all strands relative to reference strand s_ref.

    For s_ref:
      - representative residue = nearest CA in s_ref to centroid[s_ref]

    For any other strand s:
      - representative residue = residue whose CA is nearest to centroid[s_ref]

    Then for each strand:
      c1 = centroid of backbone atoms for residue r-1
      c2 = centroid of backbone atoms for residue r
      c3 = centroid of backbone atoms for residue r+1

    Args:
        pdb_path: str
        chain_id: str
        s_ref: int
        centroids: dict[int -> np.array([x,y,z])]
        c_alpha_per_strand: dict[int -> list[[x,y,z], ...]]
        strands: dict[int -> list[int]]

    Returns:
        out[strand_idx] = {
            "rep_resnum": int,
            "c1": [x,y,z] or None,
            "c2": [x,y,z] or None,
            "c3": [x,y,z] or None,
        }
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    out = {}

    # representative residue for s_ref: nearest CA to its own centroid
    nearest_self = get_nearest_c_alpha_residue_per_strand(
        centroids, c_alpha_per_strand, strands
    )
    ref_resnum = nearest_self[s_ref]["resnum"]

    for s in strands:
        if s == s_ref:
            rep_resnum = ref_resnum
        else:
            # nearest CA in strand s to centroid of s_ref
            idx_in_s, _ = get_nearest_c_alpha_to_ref_c_alpha(
                s_ref, s, centroids, c_alpha_per_strand
            )
            rep_resnum = int(strands[s][idx_in_s])

        c1 = backbone_centroid_for_residue(chain, rep_resnum - 1)
        c2 = backbone_centroid_for_residue(chain, rep_resnum)
        c3 = backbone_centroid_for_residue(chain, rep_resnum + 1)

        out[s] = {
            "rep_resnum": rep_resnum,
            "c1": None if c1 is None else c1.tolist(),
            "c2": None if c2 is None else c2.tolist(),
            "c3": None if c3 is None else c3.tolist(),
        }

    return out

#pdb_path="../output_pdbs/1A4K_L_3_107.pdb"
#pdb_path="../output_pdbs/4PB0_L_2_107.pdb"
#pdb_path="../output_pdbs/1YJD_C_3_117.pdb"
pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
chain_id = Path(pdb_path).stem.split("_")[1]
centroids = centroid_for_each_strand.centroid_per_strand_dict(pdb_path,chain_id)
c_alpha_per_strand = nearest_c_alpha.strand_coords_CA
strands = detect_strands_to_dict.result_dict[pdb_name]['strands']
triplets = get_triplet_backbone_centroids_per_strand(
    pdb_path=pdb_path,
    chain_id=chain_id,
    centroids=centroids,                  # {1: array([..]), ...}
    c_alpha_per_strand=c_alpha_per_strand,# {1: [[..],[..],..], ...}
    strands=strands                       # {1: [4,5,6], ...}
)
triplets_1 = get_triplets_relative_to_ref_strand(
    pdb_path=pdb_path,
    chain_id=chain_id,
    centroids=centroids,                  # {1: array([..]), ...}
    s_ref=1,
    c_alpha_per_strand=c_alpha_per_strand,# {1: [[..],[..],..], ...}
    strands=strands                       # {1: [4,5,6], ...}
)
all_triplets = {}

for s_ref in strands.keys():

    all_triplets[s_ref] = get_triplets_relative_to_ref_strand(
        pdb_path=pdb_path,
        chain_id=chain_id,
        centroids=centroids,
        s_ref=s_ref,
        c_alpha_per_strand=c_alpha_per_strand,
        strands=strands
    )
#print(len(triplets))
pprint.pprint(triplets)
print(len(triplets))
#print(triplets)
#pprint.pprint(all_triplets)
#print(len(all_triplets))
#print(triplets)
#print(triplets_1[1][1])
