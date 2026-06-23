
def propagate_labels_relative_to_bcef(bcef_labels, all_strand_ids):
    """
    bcef_labels:
        {strand_idx: "B"/"C"/"E"/"F"/None}

    all_strand_ids:
        list of all DSSP strand indices, e.g. [1,2,3,4,5,6,7,8,9]

    Returns:
        {strand_idx: relative_label}
    """

    strand_ids = sorted(all_strand_ids)

    # keep only known BCEF anchor labels
    anchors = {
        s: label
        for s, label in bcef_labels.items()
        if label in {"B", "C", "E", "F"}
    }

    # reverse lookup: label -> strand index
    label_to_idx = {label: s for s, label in anchors.items()}

    required = ["B", "C", "E", "F"]
    missing = [x for x in required if x not in label_to_idx]
    if missing:
        raise ValueError(f"Missing BCEF anchor labels: {missing}")

    B = label_to_idx["B"]
    C = label_to_idx["C"]
    E = label_to_idx["E"]
    F = label_to_idx["F"]

    updated = {}

    for s in strand_ids:
        if s in anchors:
            updated[s] = anchors[s]

        elif s < B:
            updated[s] = f"B-{B - s}"

        elif C < s < E:
            updated[s] = f"C+{s - C}"

        elif s > F:
            updated[s] = f"F+{s - F}"

        elif B < s < C:
            updated[s] = f"B+{s - B}"

        elif E < s < F:
            updated[s] = f"E+{s - E}"

        else:
            updated[s] = None

    return updatedlt.show()

def rule_for_G_strands(D, init_labels):
    """
    D: distance matrix, 0-indexed numpy array
    init_labels: dict like
        {
            1: "B-1",
            2: "B",
            3: "C",
            ...
            8: "F+1",
            9: "F+2"
        }

    Returns:
        updated_labels
    """

    labels = init_labels.copy()

    # find strand indices by label
    label_to_strand = {v: k for k, v in labels.items()}

    F = label_to_strand["F"]
    F_plus_1 = label_to_strand.get("F+1")
    F_plus_2 = label_to_strand.get("F+2")

    if F_plus_1 is None:
        return labels

    # F+1 becomes G
    labels[F_plus_1] = "G"

    if F_plus_2 is None:
        return labels

    # compare distances:
    # D is 0-indexed, strand labels are 1-indexed
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
    Applies only these rules:

    D[B-1,B] < D[B-1,F+1] => B-1 = A
    D[B-1,B] > D[B-1,F+1] => B-1 = A'

    D[B-2,B] < D[B-2,B-1] => B-2 = A
    D[B-2,B] > D[B-2,B-1] => B-2 = A-1
    """

    labels = labels.copy()
    label_to_strand = {v: k for k, v in labels.items()}

    B = label_to_strand.get("B")
    B_minus_1 = label_to_strand.get("B-1")
    B_minus_2 = label_to_strand.get("B-2")
    F_plus_1 = label_to_strand.get("F+1")

    # Rule for B-1
    if B is not None and B_minus_1 is not None and F_plus_1 is not None:
        d_to_B = D[B_minus_1 - 1, B - 1]
        d_to_Fplus1 = D[B_minus_1 - 1, F_plus_1 - 1]

        if d_to_B < d_to_Fplus1:
            labels[B_minus_1] = "A"
        elif d_to_B > d_to_Fplus1:
            labels[B_minus_1] = "A'"

    # Rule for B-2
    if B is not None and B_minus_2 is not None and B_minus_1 is not None:
        d_to_B = D[B_minus_2 - 1, B - 1]
        d_to_Bminus1 = D[B_minus_2 - 1, B_minus_1 - 1]

        if d_to_B < d_to_Bminus1:
            labels[B_minus_2] = "A"
        elif d_to_B > d_to_Bminus1:
            labels[B_minus_2] = "A-1"

    return labels
def rule_for_between_C_and_E(D, labels):
    """
    Applies rules for strands after C and before E.

    C+1:
      D[C+1,C] < D[C+1,E] => C+1 = C'
      D[C+1,C] > D[C+1,E] => C+1 = D

    C+2:
      D[C+2,C+1] < D[C+2,E] => C+2 = C''
      D[C+2,C+1] > D[C+2,E] => C+2 = D
    """

    labels = labels.copy()
    label_to_strand = {v: k for k, v in labels.items()}

    C = label_to_strand.get("C")
    E = label_to_strand.get("E")
    C_plus_1 = label_to_strand.get("C+1")
    C_plus_2 = label_to_strand.get("C+2")

    if C is not None and E is not None and C_plus_1 is not None:
        d_to_C = D[C_plus_1 - 1, C - 1]
        d_to_E = D[C_plus_1 - 1, E - 1]

        if d_to_C < d_to_E:
            labels[C_plus_1] = "C'"
        elif d_to_C > d_to_E:
            labels[C_plus_1] = "D"

    if C_plus_2 is not None and C_plus_1 is not None and E is not None:
        d_to_Cplus1 = D[C_plus_2 - 1, C_plus_1 - 1]
        d_to_E = D[C_plus_2 - 1, E - 1]

        if d_to_Cplus1 < d_to_E:
            labels[C_plus_2] = "C''"
        elif d_to_Cplus1 > d_to_E:
            labels[C_plus_2] = "D"

    return labels


def assign_ig_labels(D, init_labels):
    """
    Apply all Ig strand naming rules.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix.

    init_labels : dict
        Initial labels from BCEF propagation, e.g.

        {
            1: "B-2",
            2: "B-1",
            3: "B",
            4: "C",
            5: "C+1",
            6: "E",
            7: "F",
            8: "F+1",
            9: "F+2"
        }

    Returns
    -------
    dict
        Updated strand labels.
    """

    labels = init_labels.copy()

    labels = rule_for_G_strands(D, labels)
    labels = rule_for_A_strands(D, labels)
    labels = rule_for_between_C_and_E(D, labels)

    return labels
