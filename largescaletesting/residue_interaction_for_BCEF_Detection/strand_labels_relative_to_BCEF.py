import sys
from pathlib import Path

import dssp_to_bcef
import detect_strands_to_dict
import distance_rep_point_per_strand


def build_label_to_strand(labels):
    """
    Build a reverse mapping from label to DSSP strand index.

    Supports both ordinary labels:

        {4: "C+1"}

    and shared positional labels:

        {4: ("C+1", "E-1")}

    Returns
    -------
    dict
        Example:

        {
            "C+1": 4,
            "E-1": 4,
        }
    """
    label_to_strand = {}

    for strand_id, label_value in labels.items():
        if label_value is None:
            continue

        if isinstance(label_value, (tuple, list, set)):
            for label in label_value:
                label_to_strand[label] = strand_id
        else:
            label_to_strand[label_value] = strand_id

    return label_to_strand


def propagate_labels_relative_to_bcef(
    bcef_labels,
    all_strand_ids,
):
    """
    Assign positional labels relative to the BCEF anchor strands.

    Parameters
    ----------
    bcef_labels : dict
        {
            strand_idx: "B" / "C" / "E" / "F" / None
        }

    all_strand_ids : iterable
        All DSSP strand indices.

    Returns
    -------
    dict
        {
            strand_idx: positional label
        }

    A strand immediately between C and E is assigned both:

        ("C+1", "E-1")
    """
    strand_ids = sorted(all_strand_ids)

    anchors = {
        strand_id: label
        for strand_id, label in bcef_labels.items()
        if label in {"B", "C", "E", "F"}
    }

    label_to_idx = {
        label: strand_id
        for strand_id, label in anchors.items()
    }

    required = ["B", "C", "E", "F"]

    missing = [
        label
        for label in required
        if label not in label_to_idx
    ]

    if missing:
        raise ValueError(
            f"Missing BCEF anchor labels: {missing}"
        )

    B = label_to_idx["B"]
    C = label_to_idx["C"]
    E = label_to_idx["E"]
    F = label_to_idx["F"]

    updated = {}
    E_minus_1 = E - 1

    for strand_id in strand_ids:
        if strand_id in anchors:
            updated[strand_id] = anchors[strand_id]

        elif strand_id < B:
            updated[strand_id] = f"B-{B - strand_id}"

        elif B < strand_id < C:
            updated[strand_id] = f"B+{strand_id - B}"

        elif C < strand_id < E_minus_1:
            updated[strand_id] = f"C+{strand_id - C}"

        elif strand_id == E_minus_1:
            if strand_id == C + 1:
                updated[strand_id] = ("C+1", "E-1")
            else:
                updated[strand_id] = "E-1"

        elif E < strand_id < F:
            updated[strand_id] = f"E+{strand_id - E}"

        elif strand_id > F:
            updated[strand_id] = f"F+{strand_id - F}"

        else:
            updated[strand_id] = None

    return updated


def rule_for_G_strands(D, init_labels):
    """
    Apply the G-strand labeling rules.

    Rules
    -----
    F+1 becomes G.

    For F+2:

        D[F+2, F] < D[F+2, F+1]  -> G'
        D[F+2, F] > D[F+2, F+1]  -> G+1
    """
    labels = init_labels.copy()
    label_to_strand = build_label_to_strand(labels)

    F = label_to_strand.get("F")
    F_plus_1 = label_to_strand.get("F+1")
    F_plus_2 = label_to_strand.get("F+2")

    if F is None or F_plus_1 is None:
        return labels

    labels[F_plus_1] = "G"

    if F_plus_2 is None:
        return labels

    dist_to_F = D[F_plus_2 - 1, F - 1]
    dist_to_G = D[F_plus_2 - 1, F_plus_1 - 1]

    if dist_to_F < dist_to_G:
        labels[F_plus_2] = "G'"

    elif dist_to_F > dist_to_G:
        labels[F_plus_2] = "G+1"

    else:
        labels[F_plus_2] = "G?"

    return labels


def rule_for_A_strands(D, labels):
    """
    Apply the A-strand labeling rules.

    Rules
    -----
    D[B-1, B] < D[B-1, F+1]  -> B-1 = A
    D[B-1, B] > D[B-1, F+1]  -> B-1 = A'

    D[B-2, B] < D[B-2, B-1]  -> B-2 = A
    D[B-2, B] > D[B-2, B-1]  -> B-2 = A-1
    """
    labels = labels.copy()
    label_to_strand = build_label_to_strand(labels)

    B = label_to_strand.get("B")
    B_minus_1 = label_to_strand.get("B-1")
    B_minus_2 = label_to_strand.get("B-2")
    F_plus_1 = label_to_strand.get("F+1")

    if (
        B is not None
        and B_minus_1 is not None
        and F_plus_1 is not None
    ):
        dist_to_B = D[B_minus_1 - 1, B - 1]

        dist_to_F_plus_1 = D[
            B_minus_1 - 1,
            F_plus_1 - 1,
        ]

        if dist_to_B < dist_to_F_plus_1:
            labels[B_minus_1] = "A"

        elif dist_to_B > dist_to_F_plus_1:
            labels[B_minus_1] = "A'"

    if (
        B is not None
        and B_minus_2 is not None
        and B_minus_1 is not None
    ):
        dist_to_B = D[B_minus_2 - 1, B - 1]

        dist_to_B_minus_1 = D[
            B_minus_2 - 1,
            B_minus_1 - 1,
        ]

        if dist_to_B < dist_to_B_minus_1:
            labels[B_minus_2] = "A"

        elif dist_to_B > dist_to_B_minus_1:
            labels[B_minus_2] = "A-1"

    return labels


def rule_for_between_C_and_E(D, labels):
    """
    Apply labeling rules to strands between C and E.

    Rules
    -----
    C+1:

        D[C+1, C] < D[C+1, E] -> C'
        D[C+1, C] > D[C+1, E] -> D

    C+2:

        D[C+2, C+1] < D[C+2, E] -> C''
        D[C+2, C+1] > D[C+2, E] -> D
    """
    labels = labels.copy()
    label_to_strand = build_label_to_strand(labels)

    C = label_to_strand.get("C")
    E = label_to_strand.get("E")
    C_plus_1 = label_to_strand.get("C+1")
    C_plus_2 = label_to_strand.get("C+2")

    if (
        C is not None
        and E is not None
        and C_plus_1 is not None
    ):
        dist_to_C = D[C_plus_1 - 1, C - 1]
        dist_to_E = D[C_plus_1 - 1, E - 1]

        if dist_to_C < dist_to_E:
            labels[C_plus_1] = "C'"

        elif dist_to_C > dist_to_E:
            labels[C_plus_1] = "D"

    if (
        C_plus_2 is not None
        and C_plus_1 is not None
        and E is not None
    ):
        dist_to_C_plus_1 = D[
            C_plus_2 - 1,
            C_plus_1 - 1,
        ]

        dist_to_E = D[
            C_plus_2 - 1,
            E - 1,
        ]

        if dist_to_C_plus_1 < dist_to_E:
            labels[C_plus_2] = "C''"

        elif dist_to_C_plus_1 > dist_to_E:
            labels[C_plus_2] = "D"

    return labels


def assign_ig_labels(D, init_labels):
    """
    Apply all final Ig strand-labeling rules.

    Parameters
    ----------
    D : np.ndarray
        Strand-distance matrix.

    init_labels : dict
        Initial positional labels.

    Returns
    -------
    dict
        Final Ig strand labels.
    """
    labels = init_labels.copy()

    labels = rule_for_A_strands(D, labels)
    labels = rule_for_G_strands(D, labels)
    labels = rule_for_between_C_and_E(D, labels)

    return labels


def get_init_labels(
    pdb_file,
    bcef_file,
    chain_id=None,
    min_len=3,
):
    """
    Generate positional BCEF-relative labels for one protein.

    This is the function needed by the neighboring-feature pipeline.

    Parameters
    ----------
    pdb_file : str or Path
        Original PDB file.

    bcef_file : str or Path
        Matching BCEF PDB file.

    chain_id : str, optional
        Chain ID. When omitted, infer it from filenames such as:

            6UDJ_E_3_107.pdb

    min_len : int
        Minimum DSSP strand length.

    Returns
    -------
    dict
        Initial positional strand labels.
    """
    pdb_file = str(pdb_file)
    bcef_file = str(bcef_file)

    if chain_id is None:
        filename_parts = Path(pdb_file).stem.split("_")

        if len(filename_parts) < 2:
            raise ValueError(
                "Could not infer chain ID from filename "
                f"{Path(pdb_file).name!r}. "
                "Pass chain_id explicitly."
            )

        chain_id = filename_parts[1]

    dssp_result = (
        detect_strands_to_dict.detect_strands_to_dict(
            pdb_file,
            chain_id=chain_id,
            min_len=min_len,
        )
    )

    bcef_ranges = dssp_to_bcef.parse_bcef_sheet_ranges(
        bcef_file
    )

    bcef_labels = dssp_to_bcef.label_dssp_strands(
        dssp_result=dssp_result,
        bcef_ranges=bcef_ranges,
    )

    return propagate_labels_relative_to_bcef(
        bcef_labels=bcef_labels,
        all_strand_ids=bcef_labels.keys(),
    )


def get_final_labels(
    pdb_file,
    bcef_file,
    chain_id=None,
    min_len=3,
):
    """
    Generate final Ig labels for one protein.

    This function computes the distance matrix explicitly instead of using
    the old global distance_rep_point_per_strand.D value.
    """
    init_labels = get_init_labels(
        pdb_file=pdb_file,
        bcef_file=bcef_file,
        chain_id=chain_id,
        min_len=min_len,
    )

    D = (
        distance_rep_point_per_strand
        .get_distance_matrix_for_pdb(
            pdb_path=pdb_file,
            chain_id=chain_id,
            min_len=min_len,
        )
    )

    return assign_ig_labels(
        D=D,
        init_labels=init_labels,
    )


def main():
    if len(sys.argv) not in {3, 4}:
        print(
            "Usage: python strand_labels_relative_to_BCEF.py "
            "<pdb_file> <bcef_file> [chain_id]"
        )
        sys.exit(1)

    pdb_file = sys.argv[1]
    bcef_file = sys.argv[2]

    chain_id = (
        sys.argv[3]
        if len(sys.argv) == 4
        else None
    )

    init_labels = get_init_labels(
        pdb_file=pdb_file,
        bcef_file=bcef_file,
        chain_id=chain_id,
        min_len=3,
    )

    #print(f"Init labels: {init_labels}")

    final_labels = get_final_labels(
        pdb_file=pdb_file,
        bcef_file=bcef_file,
        chain_id=chain_id,
        min_len=3,
    )

    #print(f"Final labels: {final_labels}")


if __name__ == "__main__":
    main()
