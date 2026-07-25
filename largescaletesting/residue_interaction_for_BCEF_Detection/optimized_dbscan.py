import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from custom_dbi import dbi
from hdbscan.validity import validity_index


def score_dbscan(D, eps, min_samples, alpha=1.0, beta=0.01,gamma=1.0):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = model.fit_predict(D)
    n_clusters = len(set(labels) - {-1})
    if n_clusters <2:
        return -np.inf, labels, None, None, None

    mask = labels != -1
    D_core = D[np.ix_(mask, mask)]
    labels_core = labels[mask]

    noise_fraction = np.mean(labels == -1)

    try:
        dbi_score = dbi(D_core, labels_core)

        sil_score = silhouette_score(
            D_core,
            labels_core,
            metric="precomputed"
        )
        dbcv_score = validity_index(
            D_core,
            labels_core,
            metric="precomputed",
            d=3
        )

    except Exception as e:
        print("Exception:", repr(e))
        raise

    score = (
        -alpha * dbi_score +
        beta * sil_score +
        gamma * dbcv_score
        - 0.1 * noise_fraction
    )

    return score, labels, dbi_score, sil_score, dbcv_score


def tune_dbscan(
    D,
    eps_values,
    min_samples_values,
    alpha=1.0,
    beta=1,
):
    best = {
        "score": -np.inf,
        "eps": None,
        "min_samples": None,
        "labels": None,
        "dbi": None,
        "silhouette": None,
        "dbcv": None,
    }

    for eps in eps_values:
        #for min_samples in min_samples_values:
            score, labels, dbi_score, sil_score, dbcv_score = score_dbscan(
                D,
                eps,
                min_samples=3,
                alpha=alpha,
                beta=beta,
            )

            if np.isfinite(score) and score > best["score"]:
                best.update({
                    "score": score,
                    "eps": eps,
                    #"min_samples": min_samples,
                    "min_samples": 3,
                    "labels": labels,
                    "dbi": dbi_score,
                    "silhouette": sil_score,
                    "dbcv": dbcv_score,
                })

    if best["eps"] is None:
        raise ValueError(
            "No valid DBSCAN parameters found. "
            "Every run produced fewer than 2 clusters or failed scoring."
        )

    return best
