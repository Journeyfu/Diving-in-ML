# Generalized LVQ
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class GRLVQ:
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.01, n_epochs=100, phi="identity", random_state=0):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.phi = phi
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
    
    def softmax(self, x, axis=0):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

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

    
        self.alpha_array = np.ones(X.shape[1])

        for epoch in range(self.n_epochs):
            eta = self.learning_rate * (1 - epoch / self.n_epochs)
            order = self.rng.permutation(len(X))

            for i in order:
                xi, yi = X[i], y[i]

                
                lambda_array = self.softmax(self.alpha_array, axis=0)
                dists = np.sum(lambda_array * (self.prototypes - xi) ** 2, axis=1)


                mask_pos = (self.prototype_labels == yi)
                mask_neg = ~mask_pos
                idx_p = np.argmin(np.where(mask_pos, dists, np.inf))
                idx_n = np.argmin(np.where(mask_neg, dists, np.inf))

                dp, dn = dists[idx_p], dists[idx_n]
                S = dp + dn
                if S < 1e-12:
                    continue

                diff = (dp - dn) / S
                phi_grad = self.get_phi_grad(diff)

                self.prototypes[idx_p] += eta * phi_grad * (4 * dn / S ** 2) * lambda_array * (xi - self.prototypes[idx_p])
                self.prototypes[idx_n] += eta * phi_grad * (4 * dp / S ** 2) * lambda_array * (self.prototypes[idx_n] - xi)


                # oaw loss with respect to alpha

                pos_j = (xi - self.prototypes[idx_p]) ** 2
                neg_j = (xi - self.prototypes[idx_n]) ** 2

                dE_dlambda = phi_grad * 2 * (dn * pos_j - dp * neg_j) / (S ** 2)

                # chain rule through softmax
                dE_dalpha = lambda_array * (dE_dlambda - np.dot(dE_dlambda, lambda_array))

                self.alpha_array -= eta * dE_dalpha

        return self

    def predict(self, X):
        diff = X[:, None] - self.prototypes[None, :]
        dmat = np.einsum("nmd, d, nmd -> nm", diff, self.softmax(self.alpha_array), diff)
        idx = np.argmin(dmat, axis=1)
        return self.prototype_labels[idx]





if __name__ == "__main__":
    true_n_clusters = 10
    X, y_true = make_blobs(
        n_samples=2000, centers=true_n_clusters, cluster_std=2, random_state=42, shuffle=True
    )

    model = GRLVQ(n_prototypes_per_class=3, learning_rate=0.01, n_epochs=20)
    model.fit(X, y_true)
    y_pred = model.predict(X)

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("GT")
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap="jet")
    plt.title("GRLVQ Clustering")
    plt.show()
    print("results of lambda array: ",  model.softmax(model.alpha_array))
    # results of lambda array:  [0.58039342 0.41960658]