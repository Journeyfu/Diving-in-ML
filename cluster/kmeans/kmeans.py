
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


class MyKmeans:
    def __init__(self, n_clusters=3, n_iters=100, random_state=0, init='random'):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None # shape: n_clusters, d
        self.n_iters = n_iters
        self.init = init
        self.rng = np.random.default_rng(random_state)


    def _init(self, X):
        if self.init == "random":
            self.cluster_centers_ = X[self.rng.choice(X.shape[0], size=self.n_clusters, replace=False)]

        elif self.init == "k-means++":

            for i in range(self.n_clusters):
                if i == 0: # random
                    self.cluster_centers_ = X[self.rng.integers(0, X.shape[0])].reshape(1, -1)
                else:
                    closest_dist_square = ((X[:, None] - self.cluster_centers_[None])**2).sum(axis=-1).min(axis=1)
                    p = closest_dist_square / (closest_dist_square.sum() + 1e-12)
                    idx = self.rng.choice(X.shape[0], p=p)
                    self.cluster_centers_ = np.vstack([self.cluster_centers_, X[idx]])
            
        self.start_status = self.cluster_centers_.copy()

    def fit(self, X):
        self._init(X)

        for i in range(self.n_iters):
            y_hat = self.predict(X)
            new_centers = []

            for k in range(self.n_clusters):
                if np.any(y_hat == k):
                    new_centers.append(X[y_hat == k].mean(axis=0))
                else:
                    new_centers.append(X[self.rng.integers(0, X.shape[0])])
            new_centers = np.array(new_centers)

            if np.all(np.abs(new_centers - self.cluster_centers_) < 1e-6):
                print(f"Converged at iteration {i}")
                break
            self.cluster_centers_ = new_centers
    
    def predict(self, X): # X shape: N, d
        diff = (X[:, None] - self.cluster_centers_[None])**2
        diff = diff.sum(axis=-1)
        return diff.argmin(axis=1) # return the index of the closest center for each point





if __name__ == "__main__":
    
    true_n_clusters = 10
    X, y_true = make_blobs(
        n_samples=2000, centers=true_n_clusters, cluster_std=2, random_state=42
    )
    plt.subplot(1, 4, 1)

    plt.title("GT")
    for k in range(true_n_clusters):
        plt.scatter(X[y_true == k, 0], X[y_true == k, 1], s=10)

    m1 = MyKmeans(n_clusters=10, random_state=42, init="random")
    m2 = MyKmeans(n_clusters=10, random_state=42, init="k-means++")
    m3 = KMeans(n_clusters=10, random_state=42)
    
    for idx, (kmeans_obj, string) in enumerate(zip([m1, m2, m3], ["MyKmeans", "MyKmeans++", "scikit-learn"])):
        kmeans_obj.fit(X)
        y_kmeans = kmeans_obj.predict(X)
        
        inertia = np.sum((X - kmeans_obj.cluster_centers_[y_kmeans])**2)
        sil_score = silhouette_score(X, y_kmeans)
        ari = adjusted_rand_score(y_true, y_kmeans)
        nmi = normalized_mutual_info_score(y_true, y_kmeans)
        print(f"KMeans results ({string}):")
        print(f"Inertia (SSE): {inertia:.3f}" )
        print(f"Silhouette Score: {sil_score:.3f}")
        print(f"Adjusted Rand Index (ARI): {ari:.3f}")
        print(f"Normalized Mutual Info (NMI): {nmi:.3f}")
    
        plt.subplot(1, 4, idx + 2)
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=10, cmap="viridis")
    
        centers = kmeans_obj.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c="red", s=100, alpha=0.75, marker="X")

        if hasattr(kmeans_obj, 'start_status'): 
            start_status = kmeans_obj.start_status
            plt.scatter(start_status[:, 0], start_status[:, 1], c="pink", s=20, marker="*")
    
        plt.title(f"{string}", fontsize=10)
    plt.tight_layout()
    plt.show()

