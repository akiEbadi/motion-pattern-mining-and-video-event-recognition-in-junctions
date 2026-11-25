"""K-means clustering in NumPy (thesis approach 3). Used as a baseline and to start the HMM."""
import numpy as np


def kmeans_numpy(X, k=4, max_iter=200, tol=1e-5, random_state=42):
    """A small NumPy implementation of K-Means (thesis approach 3, used as a baseline)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < k:
        raise ValueError('X must be 2D and contain at least k samples')

    # Start with actual samples as centroids. The seed keeps runs repeatable.
    rng = np.random.default_rng(random_state)
    centers = X[rng.choice(len(X), k, replace=False)].copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new = centers.copy()
        for j in range(k):
            points = X[labels == j]
            if len(points):
                new[j] = points.mean(axis=0)
            else:
                # Empty clusters: move the centroid to the sample farthest from its centroid.
                new[j] = X[np.argmax(distances.min(axis=1))]

        shift = np.linalg.norm(new - centers)
        centers = new
        if shift < tol:
            break

    labels = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
    return labels, centers
