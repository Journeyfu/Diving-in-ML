import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

class MyKernelKMeans():
    def __init__(self, n_clusters=3, n_iters=100, random_state=0, gamma=None):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None # not used in kernel k-means

        self.gamma = 1.0 / X.shape[1] if gamma is None else gamma
            
        self.n_iters = n_iters
        self.rng = np.random.default_rng(random_state)
    
    def _rbf_kernel(self, X, Y):
        diff = X[:, None] - Y[None]
        return np.exp(-self.gamma * (diff**2).sum(axis=-1))

    def _init(self, X):
        K = self._rbf_kernel(X, X)
        self.train_X = X

        seeds = self.rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        self.labels = -np.ones(X.shape[0], dtype=int) 
        self.labels[seeds] = np.arange(self.n_clusters)  

        for i in range(X.shape[0]):
            if self.labels[i] != -1:
                continue
            # single element in each cluster
            d_list = []
            for k in range(self.n_clusters):
                dist = K[i, i] - 2 * K[i, seeds[k]] + 1. * K[seeds[k], seeds[k]]
                d_list.append(dist)
            self.labels[i] = np.argmin(d_list)

        return K
    
    def fit(self, X):   
        K = self._init(X)

        dist = np.zeros((X.shape[0], self.n_clusters))

        for it in range(self.n_iters):
            labels_old = self.labels.copy()
            for c_k in range(self.n_clusters):
                # x-x, x-c, c-c
                idx_k = np.where(self.labels == c_k)[0]
                dist_xx = K.diagonal() 
                dist_cc = K[np.ix_(idx_k, idx_k)].sum() / (len(idx_k)**2) 
                dist_xc = K[:, idx_k].sum(axis=1) / len(idx_k) 

                dist[:, c_k] = dist_xx - 2 * dist_xc + dist_cc

            self.labels = dist.argmin(axis=1)
            if np.all(labels_old == self.labels):
                print(f"Converged at iteration {it}")
                break
            
            n_samples_per_cluster = np.array([np.sum(self.labels == k) for k in range(self.n_clusters)])

            if np.any(n_samples_per_cluster == 0):
                empty_clusters = np.where(n_samples_per_cluster == 0)[0]
                for empty_cluster in empty_clusters:
                    idx = self.rng.choice(X.shape[0])
                    self.labels[idx] = empty_cluster
                
    def predict(self, X):
        K_xx = self._rbf_kernel(X, X)
        K_xc = self._rbf_kernel(X, self.train_X)
        K_cc = self._rbf_kernel(self.train_X, self.train_X)

        dist = np.zeros((X.shape[0], self.n_clusters))

        for c_k in range(self.n_clusters):
            # x-x, x-c, c-c
            idx_k = np.where(self.labels == c_k)[0]
            dist_xx = K_xx.diagonal()
            dist_cc = K_cc[np.ix_(idx_k, idx_k)].sum() / (len(idx_k)**2) 
            dist_xc = K_xc[:, idx_k].sum(axis=1) / len(idx_k) 

            dist[:, c_k] = dist_xx - 2 * dist_xc + dist_cc

        return dist.argmin(axis=1)

if __name__ == "__main__":
    
    normal_X, normal_y_true = make_blobs(n_samples=2000, centers=2, cluster_std=0.60, random_state=0)
    moon_X, moon_y_true = make_moons(n_samples=2000, noise=0., random_state=42)
    circles_X, circles_y_true = make_circles(n_samples=2000, noise=0.05, factor=0.2, random_state=42)

    for X, y, gamma in zip([normal_X, moon_X, circles_X], [normal_y_true, moon_y_true, circles_y_true], [0.1, 5, 10]):
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=10)
        plt.title("GT")

        kmeans = MyKernelKMeans(n_clusters=2, random_state=42, gamma=gamma)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)

        silhouette_avg = silhouette_score(X, y_kmeans)
        ari = adjusted_rand_score(y, y_kmeans)
        nmi = normalized_mutual_info_score(y, y_kmeans)

        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Normalized Mutual Information: {nmi:.4f}")
        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=10)
        plt.title("Kernel K-Means")
        plt.show()