# online kmeans clustering (competitive learning)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from cluster.kmeans.kmeans import MyKmeans
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score



class OnlineKMeansWithWarmup:
    def __init__(self, n_clusters=1, lr=0.5, random_state=0, warmup={"enable": False}):
        self.n_clusters = n_clusters
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state
        self.cluster_centers_ = None
        self.counts = None
        self.lr = lr
        self.warmup = warmup
        self.buffer = []
    
    def one_sample_fit_and_predict(self, X): 
        assert X.shape[0] == 1, "X should be a single sample, 1 x n_features"

        if self.warmup["enable"] and len(self.buffer) < self.warmup["warmup_samples"]:
            self.buffer.append(X)
            if len(self.buffer) == self.warmup["warmup_samples"]: # use kmeans++ to initialize cluster centers
                # model = MyKmeans(n_clusters=self.n_clusters, n_iters=10, init="k-means++", random_state=self.random_state)
                model = KMeans(n_clusters=self.n_clusters, init="k-means++", random_state=self.random_state, n_init=10) # more accurate than my implementation
                model.fit(np.vstack(self.buffer))
                
                # remap cluster centers and counts
                self.old_cluster_centers_ = self.cluster_centers_.copy()
                self.cluster_centers_ = self.align_centers_to_previous(self.old_cluster_centers_, model.cluster_centers_)
                self.counts = np.ones(self.n_clusters)
                print("Warmup done, re-initialized cluster centers using {} samples with k-means++".format(self.warmup["warmup_samples"]))

        if self.cluster_centers_ is None:
            self.cluster_centers_ = X.copy()
            self.counts = np.ones(1)
            return 0
        elif len(self.cluster_centers_) < self.n_clusters:
            self.cluster_centers_ = np.vstack([self.cluster_centers_, X])
            self.counts = np.append(self.counts, 1)
            return len(self.cluster_centers_) - 1

        pred = self.predict(X)
        self.counts[pred] += 1

        eta = self.lr / (1 + 0.1 * self.counts[pred]) # learning rate ( or use self.lr / np.sqrt(self.counts[pred]) )
        self.cluster_centers_[pred] += eta * (X.squeeze(axis=0) - self.cluster_centers_[pred])

        return pred.item()
            

    def predict(self, X):
        dist = np.linalg.norm(X[:, None] - self.cluster_centers_[None], axis=-1)
        return np.argmin(dist, axis=1)
    

    def align_centers_to_previous(self, old_centers, new_centers):
        C = np.linalg.norm(new_centers[:, None, :] - old_centers[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(C)  # new_i ↔ old_j

        aligned_centers = np.zeros_like(old_centers)
        for new_i, old_j in zip(row_ind, col_ind):
            aligned_centers[old_j] = new_centers[new_i]

        return aligned_centers



if __name__ == "__main__":
    true_n_clusters = 6
    show_iter_process = False
    X, y_true = make_blobs(
        n_samples=500, centers=true_n_clusters, cluster_std=1.0, random_state=43, shuffle=True
    )

    all_y_pred = []
    x_lim = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    y_lim = (X[:, 1].min() - 1, X[:, 1].max() + 1)

    model = OnlineKMeansWithWarmup(n_clusters=true_n_clusters, random_state=42, 
        warmup={"enable": True, "warmup_samples": true_n_clusters * 10})

    if show_iter_process:
        plt.ion()


    for i in range(len(X)):
        x, y = X[i:i+1], y_true[i:i+1].item()
        y_pred = model.one_sample_fit_and_predict(x)
        all_y_pred.append(y_pred)

        if show_iter_process:
            # visualization
            plt.subplot(1, 2, 1)
            plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
            plt.title("GT")

            plt.subplot(1, 2, 2)
            plt.scatter(x[:, 0], x[:, 1], c=y_pred, s=8, cmap="jet")
            center = model.cluster_centers_
            plt.scatter(center[:, 0], center[:, 1], c="red", s=50, marker="+")
            if i > 0: # visualize previous points in gray
                plt.scatter(X[:i, 0], X[:i, 1], c=all_y_pred[:-1], s=5, alpha=0.3, cmap="jet")
                plt.scatter(history_centers[:, 0], history_centers[:, 1], c="gray", s=50, marker="x", alpha=0.5)

            history_centers = center.copy()

            plt.title("Online Kmeans Clustering(iter {} / {})".format(i+1, len(X)))
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.show()
            plt.pause(0.1)
            plt.clf()


    all_y_pred = np.array(all_y_pred)

    plt.ioff()
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("GT")

    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c="gray", s=10, cmap="jet")
    plt.scatter(model.old_cluster_centers_[:, 0], model.old_cluster_centers_[:, 1], c=np.arange(true_n_clusters), s=50, marker="*", cmap="jet")
    plt.title("Online Center Before Warmup", fontsize=8)

    plt.subplot(1, 3, 3)

    plt.scatter(X[:, 0], X[:, 1], c=all_y_pred, s=10, cmap="jet")
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="red", s=50, marker="x")
    plt.title("Online KmeansClustering", fontsize=8)
    plt.show(block=True)

    inertia = np.sum((X - model.cluster_centers_[all_y_pred])**2)
    sil_score = silhouette_score(X, all_y_pred)
    ari = adjusted_rand_score(y_true, all_y_pred)
    nmi = normalized_mutual_info_score(y_true, all_y_pred)
    print(f"Online KMeans results:")
    print(f"Inertia (SSE): {inertia:.3f}" )
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.3f}")
    
    
    # Inertia (SSE): 1315.305
    # Silhouette Score: 0.479
    # Adjusted Rand Index (ARI): 0.771
    # Normalized Mutual Info (NMI): 0.816