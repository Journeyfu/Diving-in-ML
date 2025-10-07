# learning Vector Quantization (LVQ) clustering v2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# LVQ2 updates the two closest prototypes of different classes if they are within a certain window
class LVQ2:
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.01, n_epochs=100, epsilon=0.2):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.prototypes = None
        self.prototype_labels = None

        self.epsilon = epsilon
    
    def fit(self, X, y):
        self.prototypes = []
        self.prototype_labels = []

        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            idx = np.random.choice(len(X_c), self.n_prototypes_per_class, replace=False)
            self.prototypes.append(X_c[idx])
            self.prototype_labels += [c] * self.n_prototypes_per_class

        self.prototypes = np.vstack(self.prototypes)
        self.prototype_labels = np.array(self.prototype_labels)

        for epoch in range(self.n_epochs):
            eta = self.learning_rate * (1 - epoch / (self.n_epochs - 1))
            for sample, label in zip(X, y):
                # globally closest pair of different-class prototypes
                dist = np.linalg.norm(self.prototypes - sample[None], axis=-1)
                closest_idx, dist_list = self.choose_closest_different_class_prototype(dist)
                if closest_idx is None:
                    continue

                if self.decision_update(label, closest_idx, dist_list):
                    for i in range(2):
                        if self.prototype_labels[closest_idx[i]] == label:
                            self.prototypes[closest_idx[i]] += eta * (sample - self.prototypes[closest_idx[i]])
                        else:
                            self.prototypes[closest_idx[i]] -= eta * (sample - self.prototypes[closest_idx[i]])
        
    def choose_closest_different_class_prototype(self, dist):
        sorted_idx = np.argsort(dist)
        closest_idx = None
        for i in range(1, len(sorted_idx)):
            if self.prototype_labels[sorted_idx[i]] != self.prototype_labels[sorted_idx[0]]:
                closest_idx = [sorted_idx[0], sorted_idx[i]]
                break
        if closest_idx is None:
            return None, None
        dist_list = [dist[i] for i in closest_idx]
        return closest_idx, dist_list
    
    def decision_update(self, label, idx, dist):
        if min(dist) / max(dist) > (1 - self.epsilon) / (1 + self.epsilon):
            return True
        else:
            return False
        
    
    def predict(self, X):
        dist = np.linalg.norm(self.prototypes[None]-X[:, None], axis=-1)
        return self.prototype_labels[np.argmin(dist, axis=-1)]

if __name__ == "__main__":
    
    true_n_clusters = 10
    X, y_true = make_blobs(
        n_samples=2000, centers=true_n_clusters, cluster_std=2, random_state=42
    )

    model = LVQ2(n_prototypes_per_class=3, learning_rate=0.01, n_epochs=20, epsilon=0.2)
    model.fit(X, y_true)
    y_pred = model.predict(X)

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("GT")
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, alpha=0.5, cmap="jet")
    plt.scatter(model.prototypes[:, 0], model.prototypes[:, 1], c="k", s=80, marker='x', cmap="jet")
    plt.scatter(model.prototypes[:, 0], model.prototypes[:, 1], c=model.prototype_labels, s=50, marker='x', cmap="jet")
    plt.title("LVQ2 Clustering")
    plt.show()

