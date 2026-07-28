# build_neighboring_feature_matrix.py

import os
import argparse
import pandas as pd

import dssp_label_to_range_mapping

import neighboring_bcef_angles
import neighboring_bcef_angles_strand_projections
import neighboring_bcef_angles_with_axes
import neighboring_bcef_dist_from_cm
import neighboring_bcef_length_of_strands


FEATURE_FUNCTIONS = [
    neighboring_bcef_angles.calculate_features,
    neighboring_bcef_angles_strand_projections.calculate_features,
    neighboring_bcef_angles_with_axes.calculate_features,
    neighboring_bcef_dist_from_cm.calculate_features,
    neighboring_bcef_length_of_strands.calculate_features,
]

CONTINUOUS_FEATURES = [
    # 3D neighboring-strand angles: 4
    "angle_3d_Bm1_B",
    "angle_3d_C_Cp1",
    "angle_3d_Em1_E",
    "angle_3d_F_Fp1",

    # Projected neighboring-strand angles: 4
    "angle_proj_Bm1_B",
    "angle_proj_C_Cp1",
    "angle_proj_Em1_E",
    "angle_proj_F_Fp1",

    # Neighboring-strand angles with x, y, and z axes: 12
    "axis_angle_Bm1_x",
    "axis_angle_Bm1_y",
    "axis_angle_Bm1_z",

    "axis_angle_Cp1_x",
    "axis_angle_Cp1_y",
    "axis_angle_Cp1_z",

    "axis_angle_Em1_x",
    "axis_angle_Em1_y",
    "axis_angle_Em1_z",

    "axis_angle_Fp1_x",
    "axis_angle_Fp1_y",
    "axis_angle_Fp1_z",

    # Distance of neighboring-strand center of mass from z=0 plane: 4
    "cm_distance_Bm1",
    "cm_distance_Cp1",
    "cm_distance_Em1",
    "cm_distance_Fp1",

    # Geometric neighboring-strand lengths: 4
    "length_Bm1",
    "length_Cp1",
    "length_Em1",
    "length_Fp1",
]

PRESENCE_FEATURES = [
    "has_Bm1",
    "has_Cp1",
    "has_Em1",
    "has_Fp1",
]

FEATURE_COLUMNS = CONTINUOUS_FEATURES + PRESENCE_FEATURES


def strand_exists(label_coords, label):
    """
    Return 1 when the strand exists and has at least two residues worth of
    backbone coordinates (N, CA, C for each residue); otherwise return 0.
    """
    if label not in label_coords:
        return 0

    coords = label_coords[label].get("coords", [])
    return int(len(coords) >= 6)


def add_presence_features(row, label_coords):
    row["has_Bm1"] = strand_exists(label_coords, "B-1")
    row["has_Cp1"] = strand_exists(label_coords, "C+1")
    row["has_Em1"] = strand_exists(label_coords, "E-1")
    row["has_Fp1"] = strand_exists(label_coords, "F+1")
    return row


def replace_missing_values(row):
    """
    Replace missing continuous measurements with 0.0.

    Presence columns retain their binary values and indicate whether the zero
    is a true measurement or a missing-strand placeholder.
    """
    for feature in CONTINUOUS_FEATURES:
        value = row.get(feature)

        if value is None or pd.isna(value):
            row[feature] = 0.0

    return row


def compute_all_features(label_coords):
    row = {}

    for feature_function in FEATURE_FUNCTIONS:
        features = feature_function(label_coords)

        if not isinstance(features, dict):
            raise TypeError(
                f"{feature_function.__module__}.{feature_function.__name__} "
                "must return a dictionary."
            )

        duplicate_keys = set(row).intersection(features)
        if duplicate_keys:
            raise ValueError(
                "Duplicate feature names returned by feature modules: "
                f"{sorted(duplicate_keys)}"
            )

        row.update(features)

    row = add_presence_features(row, label_coords)
    row = replace_missing_values(row)

    missing_columns = [name for name in FEATURE_COLUMNS if name not in row]
    if missing_columns:
        raise ValueError(
            "Feature functions did not produce the expected columns: "
            f"{missing_columns}"
        )

    return row


def get_bcef_path(pdb_filename, bcef_dir):
    """
    Convert an original filename such as:
        6UDJ_E_3_107.pdb

    into the expected BCEF filename:
        6UDJ_E_3_107_BCEF.pdb
    """
    stem, extension = os.path.splitext(pdb_filename)

    if extension.lower() != ".pdb":
        raise ValueError(f"Expected a .pdb file, received: {pdb_filename}")

    bcef_filename = f"{stem}_BCEF.pdb"
    return os.path.join(bcef_dir, bcef_filename)


def save_failed_proteins(failed, output_csv):
    if not failed:
        return None

    output_base, _ = os.path.splitext(output_csv)
    failed_path = f"{output_base}_failed.txt"

    with open(failed_path, "w") as handle:
        for protein, error in failed:
            handle.write(f"{protein}\t{error}\n")

    return failed_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a 32-feature neighboring-strand matrix from matching "
            "original and BCEF PDB directories."
        )
    )

    parser.add_argument(
        "--pdb-dir",
        required=True,
        help="Directory containing original PDB files.",
    )

    parser.add_argument(
        "--bcef-dir",
        required=True,
        help="Directory containing files named <original_stem>_BCEF.pdb.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path of the output CSV file.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.pdb_dir):
        parser.error(f"Original PDB directory does not exist: {args.pdb_dir}")

    if not os.path.isdir(args.bcef_dir):
        parser.error(f"BCEF directory does not exist: {args.bcef_dir}")

    pdb_files = sorted(
        filename
        for filename in os.listdir(args.pdb_dir)
        if filename.lower().endswith(".pdb")
    )

    if not pdb_files:
        parser.error(f"No .pdb files found in: {args.pdb_dir}")

    output_directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_directory, exist_ok=True)

    rows = []
    failed = []
    total = len(pdb_files)

    for index, pdb_filename in enumerate(pdb_files, start=1):
        protein = os.path.splitext(pdb_filename)[0]
        pdb_path = os.path.join(args.pdb_dir, pdb_filename)
        bcef_path = get_bcef_path(pdb_filename, args.bcef_dir)

        print(f"[{index}/{total}] {protein}")

        if not os.path.isfile(bcef_path):
            message = f"Missing BCEF file: {os.path.basename(bcef_path)}"
            print(f"  FAILED: {message}")
            failed.append((protein, message))
            continue

        try:
            label_coords = dssp_label_to_range_mapping.get_label_coords_mapping(
                pdb_path,
                bcef_path,
            )

            row = {"protein": protein}
            row.update(compute_all_features(label_coords))
            rows.append(row)

        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"  FAILED: {message}")
            failed.append((protein, message))

    if not rows:
        raise RuntimeError("No proteins were processed successfully.")

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe[["protein"] + FEATURE_COLUMNS]
    dataframe.to_csv(args.output, index=False)

    failed_path = save_failed_proteins(failed, args.output)

    print()
    print(f"Successfully processed: {len(dataframe)}")
    print(f"Failed: {len(failed)}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)}")
    print(f"Saved matrix: {args.output}")

    if failed_path is not None:
        print(f"Saved failure log: {failed_path}")


if __name__ == "__main__":
    main()
