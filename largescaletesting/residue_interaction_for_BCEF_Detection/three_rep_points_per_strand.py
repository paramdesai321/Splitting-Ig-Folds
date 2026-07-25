import os
import sys
import pprint
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

import centroid_for_each_strand
import nearest_c_alpha
import detect_strands_to_dict


BACKBONE_ATOMS = ("N", "CA", "C")


def backbone_centroid_for_residue(chain, resseq: int):
    """
    Compute centroid of N, CA, C for residue with given residue number.

    Returns
    -------
    np.ndarray or None
        np.array([x, y, z]) or None if the residue/atoms are missing.
    """
    res_obj = None

    for residue in chain:
        if residue.id[0] != " ":
            continue

        if residue.id[1] == resseq:
            res_obj = residue
            break

    if res_obj is None:
        return None

    coords = []

    for atom_name in BACKBONE_ATOMS:
        if res_obj.has_id(atom_name):
            coords.append(res_obj[atom_name].get_coord())

    if not coords:
        return None

    coords = np.asarray(coords, dtype=float)
    return coords.mean(axis=0)


def get_nearest_c_alpha_residue_per_strand(
    centroids,
    c_alpha_per_strand,
    strands,
):
    """
    Find the representative residue for each strand.

    The representative residue is the residue whose C-alpha is nearest to
    the centroid of that strand.

    Parameters
    ----------
    centroids : dict
        {strand_idx: np.array([x, y, z])}

    c_alpha_per_strand : dict
        {
            strand_idx: [
                [x, y, z],
                ...
            ]
        }

    strands : dict
        {
            strand_idx: [
                residue_number,
                ...
            ]
        }

    Returns
    -------
    dict
        {
            strand_idx: {
                "resnum": int,
                "coord": [x, y, z],
                "distance": float,
            }
        }
    """
    nearest = {}

    for strand_idx in centroids:
        centroid = np.asarray(
            centroids[strand_idx],
            dtype=float,
        )

        ca_coords = np.asarray(
            c_alpha_per_strand[strand_idx],
            dtype=float,
        )

        distances = np.linalg.norm(
            ca_coords - centroid,
            axis=1,
        )

        nearest_index = int(np.argmin(distances))

        nearest[strand_idx] = {
            "resnum": int(strands[strand_idx][nearest_index]),
            "coord": ca_coords[nearest_index].tolist(),
            "distance": float(distances[nearest_index]),
        }

    return nearest


def get_nearest_c_alpha_to_ref_c_alpha(
    s_ref,
    s,
    centroids,
    c_alpha_per_strand,
):
    """
    Find the C-alpha in strand s nearest to the centroid of strand s_ref.

    Returns
    -------
    tuple
        (index_in_s, coordinate_in_s)
    """
    ref_centroid = np.asarray(
        centroids[s_ref],
        dtype=float,
    )

    target_coords = np.asarray(
        c_alpha_per_strand[s],
        dtype=float,
    )

    distances = np.linalg.norm(
        target_coords - ref_centroid,
        axis=1,
    )

    nearest_index = int(np.argmin(distances))

    return nearest_index, target_coords[nearest_index].tolist()


def get_triplet_backbone_centroids_per_strand(
    pdb_path,
    chain_id,
    centroids,
    c_alpha_per_strand,
    strands,
):
    """
    Build an independent representative triplet for every strand.

    For each strand:
      1. Select the residue whose C-alpha is nearest its own centroid.
      2. Compute backbone centroids for residue-1, residue, and residue+1.

    Returns
    -------
    dict
        {
            strand_idx: {
                "rep_resnum": int,
                "c1": [x, y, z] or None,
                "c2": [x, y, z] or None,
                "c3": [x, y, z] or None,
            }
        }
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    nearest = get_nearest_c_alpha_residue_per_strand(
        centroids,
        c_alpha_per_strand,
        strands,
    )

    output = {}

    for strand_idx, representative in nearest.items():
        residue_number = representative["resnum"]

        c1 = backbone_centroid_for_residue(
            chain,
            residue_number - 1,
        )
        c2 = backbone_centroid_for_residue(
            chain,
            residue_number,
        )
        c3 = backbone_centroid_for_residue(
            chain,
            residue_number + 1,
        )

        output[strand_idx] = {
            "rep_resnum": residue_number,
            "c1": None if c1 is None else c1.tolist(),
            "c2": None if c2 is None else c2.tolist(),
            "c3": None if c3 is None else c3.tolist(),
        }

    return output


def get_triplets_relative_to_ref_strand(
    pdb_path,
    chain_id,
    s_ref,
    centroids,
    c_alpha_per_strand,
    strands,
):
    """
    Build strand triplets relative to reference strand s_ref.

    For s_ref:
      - representative residue is the C-alpha nearest its own centroid.

    For every other strand:
      - representative residue is the C-alpha nearest centroid[s_ref].

    Returns
    -------
    dict
        {
            strand_idx: {
                "rep_resnum": int,
                "c1": [x, y, z] or None,
                "c2": [x, y, z] or None,
                "c3": [x, y, z] or None,
            }
        }
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    output = {}

    nearest_self = get_nearest_c_alpha_residue_per_strand(
        centroids,
        c_alpha_per_strand,
        strands,
    )

    ref_resnum = nearest_self[s_ref]["resnum"]

    for strand_idx in strands:
        if strand_idx == s_ref:
            representative_resnum = ref_resnum
        else:
            nearest_index, _ = get_nearest_c_alpha_to_ref_c_alpha(
                s_ref,
                strand_idx,
                centroids,
                c_alpha_per_strand,
            )

            representative_resnum = int(
                strands[strand_idx][nearest_index]
            )

        c1 = backbone_centroid_for_residue(
            chain,
            representative_resnum - 1,
        )
        c2 = backbone_centroid_for_residue(
            chain,
            representative_resnum,
        )
        c3 = backbone_centroid_for_residue(
            chain,
            representative_resnum + 1,
        )

        output[strand_idx] = {
            "rep_resnum": representative_resnum,
            "c1": None if c1 is None else c1.tolist(),
            "c2": None if c2 is None else c2.tolist(),
            "c3": None if c3 is None else c3.tolist(),
        }

    return output


def get_all_triplets_for_pdb(
    pdb_path,
    chain_id=None,
    min_len=3,
):
    """
    Detect strands and build all reference-relative triplet mappings.

    Parameters
    ----------
    pdb_path : str or Path
        Path to the original PDB file.

    chain_id : str, optional
        Chain ID. If omitted, infer it from filenames such as:
            6UDJ_E_3_107.pdb

    min_len : int
        Minimum DSSP strand length.

    Returns
    -------
    dict
        {
            reference_strand: {
                target_strand: {
                    "rep_resnum": int,
                    "c1": ...,
                    "c2": ...,
                    "c3": ...,
                }
            }
        }
    """
    pdb_path = str(pdb_path)

    if chain_id is None:
        stem_parts = Path(pdb_path).stem.split("_")

        if len(stem_parts) < 2:
            raise ValueError(
                "Could not infer chain ID from filename "
                f"{Path(pdb_path).name!r}. "
                "Pass chain_id explicitly."
            )

        chain_id = stem_parts[1]

    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")

    dssp_result = detect_strands_to_dict.detect_strands_to_dict(
        pdb_path,
        chain_id=chain_id,
        min_len=min_len,
    )

    strands = dssp_result[pdb_name]["strands"]

    centroids = centroid_for_each_strand.centroid_per_strand_dict(
        pdb_path,
        chain_id,
    )

    c_alpha_per_strand = (
        centroid_for_each_strand.parse_pdb_backbone_coords_by_strand(
            pdb_path,
            chain_id,
            strands,
            BACKBONE_ATOMS={"CA"},
        )
    )

    all_triplets = {}

    for s_ref in strands:
        all_triplets[s_ref] = get_triplets_relative_to_ref_strand(
            pdb_path=pdb_path,
            chain_id=chain_id,
            s_ref=s_ref,
            centroids=centroids,
            c_alpha_per_strand=c_alpha_per_strand,
            strands=strands,
        )

    return all_triplets


def main():
    if len(sys.argv) not in {2, 3}:
        print(
            "Usage: python three_rep_points_per_strand.py "
            "<pdb_file> [chain_id]"
        )
        sys.exit(1)

    pdb_path = sys.argv[1]
    chain_id = sys.argv[2] if len(sys.argv) == 3 else None

    all_triplets = get_all_triplets_for_pdb(
        pdb_path,
        chain_id=chain_id,
        min_len=3,
    )

    pprint.pprint(all_triplets)
    #print(len(all_triplets))


if __name__ == "__main__":
    main()
