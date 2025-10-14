# (Robust) Soft LVQ

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class RSLVQ:
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.01, n_epochs=20, random_state=42):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
    
        self.sigma = 0.5
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        classes = np.unique(y)
        self.prototypes = []
        self.prototype_labels = []

        for c in classes:
            Xc = X[y == c]
            idx = self.rng.choice(len(Xc), self.n_prototypes_per_class, replace=False)
            self.prototypes.append(Xc[idx])
            self.prototype_labels += [c] * self.n_prototypes_per_class

        self.prototypes = np.vstack(self.prototypes)
        self.prototype_labels = np.array(self.prototype_labels)
        self.prior = 1. / (len(classes) * self.n_prototypes_per_class)

        for epoch in range(self.n_epochs):
            eta = self.learning_rate * (1 - epoch / self.n_epochs)
            order = self.rng.permutation(len(X))

            for i in order:
                xi, yi = X[i], y[i]

                dist = np.sum((self.prototypes - xi) ** 2, axis=1)
                fi = self.prior * np.exp( - dist / (2 * self.sigma**2))

                p_xi = fi.sum() + 1e-12
                p_xi_yi = fi[self.prototype_labels == yi].sum() + 1e-12

                ri = fi / p_xi
                ri_pos = fi / p_xi_yi
                ri_pos[self.prototype_labels != yi] = 0
                
                self.prototypes += eta * (ri_pos - ri)[:, None]* (xi - self.prototypes) / (self.sigma**2)

        return self

    def predict(self, X):
        dmat = np.sum((X[:, None, :] - self.prototypes[None, :, :]) ** 2, axis=2)
        idx = np.argmin(dmat, axis=1)
        return self.prototype_labels[idx]

if __name__ == "__main__":
    true_n_clusters = 10
    X, y_true = make_blobs(
        n_samples=2000, centers=true_n_clusters, cluster_std=2, random_state=42, shuffle=True
    )

    model = RSLVQ(n_prototypes_per_class=3, learning_rate=0.01, n_epochs=20)
    model.fit(X, y_true)
    y_pred = model.predict(X)

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("GT")
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap="jet")
    plt.title("RSLVQ Clustering")
    plt.show()