# Generalized Matrix LVQ with Dimensionality Reduction
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class GMLVQ:
    def __init__(
        self,
        n_prototypes_per_class=1,
        learning_rate=0.01,
        n_epochs=100,
        phi="identity",
        latent_dim=None,
        random_state=0,
    ):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.phi = phi
        self.latent_dim = latent_dim  # e.g. 2 for 2D projection
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def get_phi_grad(self, x):
        if self.phi == "identity":
            return np.ones_like(x)
        elif self.phi == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1 - s)
        else:
            raise ValueError("Unknown phi function")

    def fit(self, X, y):
        n_samples, D = X.shape
        L = self.latent_dim if self.latent_dim is not None else D

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

        self.omega = np.zeros((L, D))
        np.fill_diagonal(self.omega, 1.0)

        for epoch in range(self.n_epochs):
            eta = self.learning_rate * (1 - epoch / self.n_epochs)
            order = self.rng.permutation(n_samples)

            for i in order:
                xi, yi = X[i], y[i]

                Lambda_matrix = self.omega.T @ self.omega  # (D, D)

                diff = self.prototypes - xi  # (N, D)
                dists = np.sum((diff @ Lambda_matrix) * diff, axis=1)  # (N,)

                mask_pos = (self.prototype_labels == yi)
                mask_neg = ~mask_pos
                idx_p = np.argmin(np.where(mask_pos, dists, np.inf))
                idx_n = np.argmin(np.where(mask_neg, dists, np.inf))

                dp, dn = dists[idx_p], dists[idx_n]
                S = dp + dn
                if S < 1e-12:
                    continue

                mu = (dp - dn) / S
                phi_grad = self.get_phi_grad(mu)

                v_p = xi - self.prototypes[idx_p]
                v_n = xi - self.prototypes[idx_n]

                self.prototypes[idx_p] += eta * phi_grad * (4 * dn / S**2) * (Lambda_matrix @ v_p)
                self.prototypes[idx_n] -= eta * phi_grad * (4 * dp / S**2) * (Lambda_matrix @ v_n)

                M = dn * np.outer(v_p, v_p) - dp * np.outer(v_n, v_n)
                dE_domega = phi_grad * 4 * (self.omega @ M) / S**2
                self.omega -= eta * dE_domega

            # keep ||Ω||_F = 1
            self.omega /= np.linalg.norm(self.omega, ord="fro")

        return self

    def predict(self, X):
        Lambda_matrix = self.omega.T @ self.omega
        diff = X[:, None] - self.prototypes[None, :]
        dmat = np.einsum("nmd,dd,nmd->nm", diff, Lambda_matrix, diff)
        idx = np.argmin(dmat, axis=1)
        return self.prototype_labels[idx]

    def transform(self, X):
        # project X to latent space
        return (self.omega @ X.T).T


# ------------------ 测试 ------------------
if __name__ == "__main__":
    X, y_true = make_blobs(
        n_samples=2000, centers=4, cluster_std=2, random_state=42, shuffle=True
    )

    model = GMLVQ(n_prototypes_per_class=2, learning_rate=0.01, n_epochs=30, latent_dim=2)
    model.fit(X, y_true)
    y_pred = model.predict(X)

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("Ground Truth")

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap="jet")
    plt.scatter(model.prototypes[:, 0], model.prototypes[:, 1], c="black", s=50, marker="x")
    plt.title("GMLVQ Clustering")
    plt.show()

    Lambda_matrix = model.omega.T @ model.omega
    print("Learned Λ:\n", Lambda_matrix)
    plt.imshow(Lambda_matrix, cmap="coolwarm")
    plt.colorbar(label="Λ value")
    plt.title("Learned Λ (feature relevance matrix)")
    plt.show()

    X_proj = model.transform(X)
    plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("Projected Space (ΩX)")
    plt.show()
