### NOTE: The residues include only backbone atoms.

from pathlib import Path

from detect_strands_to_dict import detect_strands_to_dict
import strand_labels_relative_to_BCEF


def get_dssp_ranges_by_label(dssp_result, labels):
    """
    Convert:

        {strand_id: label}

    into:

        {
            label: {
                "strand_id": ...,
                "start": ...,
                "end": ...,
                "residues": [...]
            }
        }

    Shared labels such as:

        ("C+1", "E-1")

    are expanded into two separate dictionary keys.
    """
    pdb_name = next(iter(dssp_result))
    strands = dssp_result[pdb_name]["strands"]

    ranges_by_label = {}

    for strand_id, label in labels.items():
        if label is None:
            continue

        residues = strands[strand_id]

        entry = {
            "strand_id": strand_id,
            "start": min(residues),
            "end": max(residues),
            "residues": residues,
        }

        if isinstance(label, (tuple, list, set)):
            for individual_label in label:
                ranges_by_label[individual_label] = entry.copy()
        else:
            ranges_by_label[label] = entry

    return ranges_by_label


def attach_backbone_coords_by_label(
    pdb_path,
    chain_id,
    label_mapping,
):
    """
    Attach ordered N, CA, C coordinates to each labeled strand.

    The same physical strand may appear under multiple labels, such as:

        C+1
        E-1

    Therefore, coordinates are appended to every matching label and there
    is intentionally no break inside the label loop.
    """
    backbone_atoms = {"N", "CA", "C"}

    updated = {
        label: {
            **info,
            "coords": [],
        }
        for label, info in label_mapping.items()
    }

    with open(pdb_path, "r") as pdb_file:
        for line in pdb_file:
            if not line.startswith("ATOM"):
                continue

            if line[21].strip() != chain_id:
                continue

            atom_name = line[12:16].strip()

            if atom_name not in backbone_atoms:
                continue

            residue_number = int(
                line[22:26].strip()
            )

            coordinate = [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]

            for label, info in updated.items():
                if residue_number in info["residues"]:
                    info["coords"].append(
                        coordinate
                    )

    return updated


def infer_chain_id(pdb_path):
    """
    Infer chain ID from filenames such as:

        6UDJ_E_3_107.pdb
    """
    filename_parts = Path(pdb_path).stem.split("_")

    if len(filename_parts) < 2:
        raise ValueError(
            "Could not infer chain ID from filename "
            f"{Path(pdb_path).name!r}."
        )

    return filename_parts[1]


def get_label_coords_mapping(
    pdb_path,
    bcef_path,
    chain_id=None,
    min_len=3,
):
    """
    Build BCEF-relative label-to-coordinate mapping for one protein.

    This function uses initial positional labels:

        B-1, B, C, C+1, E-1, E, F, F+1, ...

    Parameters
    ----------
    pdb_path : str or Path
        Original PDB file.

    bcef_path : str or Path
        Matching BCEF PDB file.

    chain_id : str, optional
        Chain ID. If omitted, infer it from the original filename.

    min_len : int
        Minimum DSSP strand length.

    Returns
    -------
    dict
        Label-to-range-and-coordinate mapping.
    """
    pdb_path = str(pdb_path)
    bcef_path = str(bcef_path)

    if chain_id is None:
        chain_id = infer_chain_id(
            pdb_path
        )

    dssp_result = detect_strands_to_dict(
        pdb_path,
        chain_id=chain_id,
        min_len=min_len,
    )

    init_labels = (
        strand_labels_relative_to_BCEF
        .get_init_labels(
            pdb_file=pdb_path,
            bcef_file=bcef_path,
            chain_id=chain_id,
            min_len=min_len,
        )
    )

    ranges = get_dssp_ranges_by_label(
        dssp_result,
        init_labels,
    )

    label_coords = attach_backbone_coords_by_label(
        pdb_path,
        chain_id,
        ranges,
    )

    return label_coords
