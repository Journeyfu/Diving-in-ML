import numpy as np
from data import get_onehot
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

class NaiveBayes():
    def __init__(self):
        self.priors = None # (nclasses,)
        self.mus = None    # (nclasses, nfeatures)
        self.sigma_square = None  # (nclasses, nfeatures)
    
    
    def predict(self, X): # X: n x d
        a = (X[:, None] - self.mus[None])**2 / (2 * self.sigma_square[None])  # (n, c, d)
        b = 0.5 * np.log(self.sigma_square)[None] # (1, c, d)
        c = np.log(self.priors)[None]

        logits = np.sum(-a - b, axis=-1) + c
        return np.argmax(logits, axis=1)

    def fit(self, X, y):
        onehot = get_onehot(y.flatten())
        nsamples = np.bincount(y.flatten()).astype(float)
        self.priors = nsamples / y.shape[0]  # (nclasses,)
        self.mus = (X.T @ onehot).T / nsamples.reshape(-1, 1)

        diff_square = (X - self.mus[y.flatten()])**2

        self.sigma_square = (diff_square.T @ onehot).T / nsamples.reshape(-1, 1)

    def plot_decision_boundary(self, X, y, h=0.02):
        if X.shape[1] != 2:
            raise ValueError("support 2D data only")

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=20, cmap=plt.cm.coolwarm)
        plt.title("Naive Bayes Boundary")
        plt.show()


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0, class_sep=3, n_clusters_per_class=1, random_state=42)
    y = y.reshape(-1, 1)

    model = NaiveBayes()
    model.fit(X, y)
    model.plot_decision_boundary(X, y)
    acc = accuracy_score(y, model.predict(X))
    print(f"Accuracy: {acc*100:.2f}%")

