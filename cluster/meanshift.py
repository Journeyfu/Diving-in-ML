
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift 

import numpy as np
from scipy.spatial.distance import cdist

class MyMeanShift:
    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-3, merge_radius_factor=1.0):
        self.bandwidth = float(bandwidth)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.merge_radius_factor = float(merge_radius_factor)

    def _rbf_kernel(self, d2):
        # d2 = squared distances
        return np.exp(-d2 / (2.0 * self.bandwidth ** 2))

    def fit_predict(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        centers = []

        for i in range(n):
            c = X[i].copy()
            for _ in range(self.max_iter):
                diff = X - c
                dist = np.linalg.norm(diff, axis=1)
                mask = dist <= self.bandwidth
                if not np.any(mask): # no points within bandwidth
                    j = np.argmin(dist)
                    mask[j] = True

                diff_n = X[mask] - c
                d2 = np.sum(diff_n * diff_n, axis=1)
                w = self._rbf_kernel(d2)
                c_new = (w[:, None] * X[mask]).sum(axis=0) / w.sum()

                if np.linalg.norm(c_new - c) < self.tol:
                    break
                c = c_new
            centers.append(c)

        centers = np.vstack(centers)

        # get peak intensities and merge close peaks
        # intensities: kernel weighted sum within bandwidth
        intensities = []
        for c in centers:
            diff = X - c
            d2 = np.sum(diff * diff, axis=1)
            mask = d2 <= (self.bandwidth ** 2)
            w = self._rbf_kernel(d2[mask])
            intensities.append(w.sum())

        order = np.argsort(-np.asarray(intensities))  # strong to weak
        keep = []
        for idx in order:
            if len(keep)==0:
                keep.append(idx)
                continue

            dists = np.linalg.norm(centers[keep] - centers[idx], axis=1)
            if np.all(dists > self.merge_radius_factor * self.bandwidth):
                keep.append(idx)

        unique_centers = centers[keep]
        self.cluster_centers_ = unique_centers

        labels = np.argmin(cdist(X, unique_centers), axis=1)
        return labels


if __name__ == "__main__":
    
    true_n_clusters = 10
    X, y_true = make_blobs(
        n_samples=2000, centers=true_n_clusters, cluster_std=1.5, random_state=42
    )
    model = MyMeanShift(bandwidth=3)
    y_meanshift = model.fit_predict(X)
    plt.subplot(1, 3, 1)
     
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("GT")
    
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_meanshift, s=10, cmap="jet")
    center = model.cluster_centers_
    plt.scatter(center[:, 0], center[:, 1], c="k", s=30, marker="x")
    plt.title("MyMeanshift")

    model = MeanShift(bandwidth=3)
    y_meanshift = model.fit_predict(X)
    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], c=y_meanshift, s=10, cmap="jet")
    plt.title("scikit-learn")
    
    plt.tight_layout()
    plt.show()