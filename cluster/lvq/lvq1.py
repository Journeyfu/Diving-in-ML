# learning Vector Quantization (LVQ) clustering
# supervised learning algorithm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
class LVQ1:
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.01, n_epochs=100):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.prototypes = None
        self.prototype_labels = None
    
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
                dist = np.linalg.norm(self.prototypes - sample[None], axis=-1)
                closest_idx = np.argmin(dist)
                if self.prototype_labels[closest_idx] == label:
                    self.prototypes[closest_idx] += eta * (sample - self.prototypes[closest_idx])
                else:
                    self.prototypes[closest_idx] -= eta * (sample - self.prototypes[closest_idx])
    
    def predict(self, X):
        dist = np.linalg.norm(self.prototypes[None]-X[:, None], axis=-1)
        return self.prototype_labels[np.argmin(dist, axis=-1)]

if __name__ == "__main__":
    
    true_n_clusters = 10
    X, y_true = make_blobs(
        n_samples=2000, centers=true_n_clusters, cluster_std=2, random_state=42
    )

    model = LVQ1(n_prototypes_per_class=3, learning_rate=0.01, n_epochs=20)
    model.fit(X, y_true)
    y_pred = model.predict(X)

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("GT")
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, alpha=0.5, cmap="jet")
    plt.scatter(model.prototypes[:, 0], model.prototypes[:, 1], c=model.prototype_labels, s=50, marker='x', cmap="jet")
    plt.title("LVQ Clustering")
    plt.show()