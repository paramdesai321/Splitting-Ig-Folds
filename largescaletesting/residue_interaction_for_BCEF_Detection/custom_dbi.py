import numpy as np
from sklearn.preprocessing import LabelEncoder


def dbi(D, labels):
    D = np.asarray(D, dtype=float)
    labels = np.asarray(labels)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix.")

    if labels.shape[0] != D.shape[0]:
        raise ValueError("labels must have length equal to D.shape[0].")

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples = D.shape[0]
    n_labels = len(np.unique(labels))

    if not 1 < n_labels < n_samples:
        raise ValueError(
            f"Number of labels is {n_labels}. Valid values are 2 to n_samples - 1."
        )

    intra_dists = np.zeros(n_labels, dtype=float)
    medoids = np.zeros(n_labels, dtype=int)

    for k in range(n_labels):
        idx = np.where(labels == k)[0]

        D_k = D[np.ix_(idx, idx)]

        medoid_local = np.argmin(D_k.mean(axis=1))
        medoid = idx[medoid_local]

        medoids[k] = medoid
        intra_dists[k] = D[idx, medoid].mean()

    medoid_distances = D[np.ix_(medoids, medoids)]

    if np.allclose(intra_dists, 0) or np.allclose(medoid_distances, 0):
        return 0.0

    medoid_distances[medoid_distances == 0] = np.inf

    combined_intra_dists = intra_dists[:, None] + intra_dists
    scores = np.max(combined_intra_dists / medoid_distances, axis=1)

    return float(np.mean(scores))
