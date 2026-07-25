import os
import sys
from pathlib import Path

import numpy as np

from centroid_for_each_strand import (
    centroid_per_strand_dict,
    parse_pdb_backbone_coords_by_strand,
)
import detect_strands_to_dict


def get_nearest_c_alpha_to_centroid_per_strand(
    centroids,
    c_alpha_per_strand,
):
    """
    Find the C-alpha coordinate nearest to each strand centroid.

    Parameters
    ----------
    centroids : dict
        {
            strand_idx: np.array([x, y, z]),
            ...
        }

    c_alpha_per_strand : dict
        {
            strand_idx: [
                [x, y, z],
                [x, y, z],
                ...
            ],
            ...
        }

    Returns
    -------
    dict
        {
            strand_idx: [x, y, z],
            ...
        }
    """

    result = {}

    for strand_idx, centroid_value in centroids.items():
        if strand_idx not in c_alpha_per_strand:
            continue

        centroid = np.asarray(centroid_value, dtype=float)
        coords = np.asarray(
            c_alpha_per_strand[strand_idx],
            dtype=float,
        )

        if len(coords) == 0:
            continue

        distances = np.linalg.norm(
            coords - centroid,
            axis=1,
        )

        nearest_index = int(np.argmin(distances))
        result[strand_idx] = coords[nearest_index].tolist()

    return result


def get_nearest_c_alpha_for_pdb(
    pdb_path,
    chain_id=None,
    min_len=3,
):
    """
    Detect strands and find the C-alpha nearest to each strand centroid.

    Parameters
    ----------
    pdb_path : str or Path
        Path to the original PDB file.

    chain_id : str, optional
        Chain to process. If omitted, it is inferred from filenames such as:
            6UDJ_E_3_107.pdb
        where the chain is E.

    min_len : int
        Minimum DSSP strand length.

    Returns
    -------
    dict
        {
            strand_idx: [x, y, z],
            ...
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

    strand_coords_ca = parse_pdb_backbone_coords_by_strand(
        pdb_path,
        chain_id,
        strands,
        BACKBONE_ATOMS={"CA"},
    )

    centroids = centroid_per_strand_dict(
        pdb_path,
        chain_id,
    )

    return get_nearest_c_alpha_to_centroid_per_strand(
        centroids,
        strand_coords_ca,
    )


def main():
    if len(sys.argv) not in {2, 3}:
        print(
            "Usage: python nearest_c_alpha.py "
            "<pdb_file> [chain_id]"
        )
        sys.exit(1)

    pdb_path = sys.argv[1]
    chain_id = sys.argv[2] if len(sys.argv) == 3 else None

    result = get_nearest_c_alpha_for_pdb(
        pdb_path,
        chain_id=chain_id,
        min_len=3,
    )

    #print(result)


if __name__ == "__main__":
    main()
