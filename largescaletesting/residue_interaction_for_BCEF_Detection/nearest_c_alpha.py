import numpy as np
from pathlib import Path

from centroid_for_each_strand import centroid_per_strand_dict, parse_pdb_backbone_coords_by_strand
import detect_strands_to_dict
import os
import sys
import numpy as np
pdb_path = sys.argv[1]
def get_nearest_c_alpha_to_centroid_per_strand(centroids, c_alpha_per_strand):
    """
    centroids: dict
        { strand_idx : np.array([x,y,z]) }

    c_alpha_per_strand: dict
        { strand_idx : [[x,y,z], [x,y,z], ...] }

    Returns:
        {
          strand_idx: {
              "coord": [x,y,z],
              "distance": float
          },
          ...
        }
    """

    result = {}

    for strand_idx in centroids.keys():

        centroid = np.array(centroids[strand_idx])
        coords = np.array(c_alpha_per_strand[strand_idx])

        # distances from centroid to all Cα in same strand
        dists = np.linalg.norm(coords - centroid, axis=1)

        min_idx = np.argmin(dists)

        result[strand_idx] = coords[min_idx].tolist()
        

    return result

#pdb_path = "../output_pdbs/1A4K_L_3_107.pdb"
#pdb_path = "../output_pdbs/4PB0_L_2_107.pdb"
#pdb_path = "../output_pdbs/1YJD_C_3_117.pdb"
chain_id = Path(pdb_path).stem.split("_")[1]
pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
strands = detect_strands_to_dict.result_dict[pdb_name]['strands']
strand_coords_CA = parse_pdb_backbone_coords_by_strand(pdb_path, chain_id, strands,BACKBONE_ATOMS={'CA'})
print(f"C alphas in each strand {strand_coords_CA}")
centroid = centroid_per_strand_dict(pdb_path,chain_id)
print(f"Centroids of Backbone in each strands {centroid}")



print(get_nearest_c_alpha_to_centroid_per_strand(centroid, strand_coords_CA))

