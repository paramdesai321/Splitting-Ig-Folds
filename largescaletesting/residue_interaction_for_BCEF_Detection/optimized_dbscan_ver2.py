import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

def distance_pseudo_f(D, labels):
    mask = labels != -1
    D = D[np.ix_(mask, mask)]
    labels = labels[mask]

    unique = np.unique(labels)
    k = len(unique)
    n = len(labels)

    if k < 2 or n <= k:
        return -np.inf

    # Total dispersion from all pairwise squared distances
    T = np.sum(D ** 2) / n

    # Within-cluster dispersion
    W = 0.0
    for c in unique:
        idx = labels == c
        D_c = D[np.ix_(idx, idx)]
        n_c = np.sum(idx)
        W += np.sum(D_c ** 2) / n_c

    B = T - W

    if W <= 0:
        return np.inf

    return (B / (k - 1)) / (W / (n - k))


def distance_dbi(D, labels):
    unique = np.unique(labels)
    k = len(unique)

    if k < 2:
        return np.inf

    scatters = {}

    for c in unique:
        idx = labels == c
        D_c = D[np.ix_(idx, idx)]
        n_c = np.sum(idx)

        if n_c <= 1:
            scatters[c] = 0.0
        else:
            scatters[c] = np.mean(D_c)

    db_values = []

    for i in unique:
        max_ratio = -np.inf

        for j in unique:
            if i == j:
                continue

            idx_i = labels == i
            idx_j = labels == j
            separation = np.mean(D[np.ix_(idx_i, idx_j)])

            if separation <= 0:
                ratio = np.inf
            else:
                ratio = (scatters[i] + scatters[j]) / separation

            max_ratio = max(max_ratio, ratio)

        db_values.append(max_ratio)

    return np.mean(db_values)


def score_dbscan(D, eps, min_samples, alpha=1.0, beta=0.01):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = model.fit_predict(D)

    n_clusters = len(set(labels) - {-1})
    if n_clusters != 2:
        return -np.inf, labels

    mask = labels != -1
    D_core = D[np.ix_(mask, mask)]
    labels_core = labels[mask]

    noise_fraction = np.mean(labels == -1)

    try:
        dbi = distance_dbi(D_core, labels_core)
        psf = distance_pseudo_f(D, labels)
    except Exception:
        return -np.inf, labels

    score = -alpha * dbi + beta * psf - 0.1 * noise_fraction
    return dbi, psf

def tune_dbscan(D, eps_values, min_samples_values, alpha=1.0, beta=0.01):
    best = {
        "score": -np.inf,
        "eps": None,
        "min_samples": None,
        "labels": None,
        "dbi": None,
        "psf": None,
    }

    for eps in eps_values:
        for min_samples in min_samples_values:
            score, labels, dbi, psf = score_dbscan(
                D, eps, min_samples, alpha=alpha, beta=beta
            )

            if score > best["score"]:
                best.update({
                    "score": score,
                    "eps": eps,
                    "min_samples": min_samples,
                    "labels": labels,
                    "dbi": dbi,
                    "psf": psf,
                })

    return best
