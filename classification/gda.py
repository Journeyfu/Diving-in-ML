import numpy as np
from data import get_onehot
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

class GDA():
    def __init__(self):
        self.priors = None
        self.mus = None
        self.sigma = None
        self.invSigma = None
    
    def fit(self, X, y):
        onehot = get_onehot(y.flatten()) # nsamples, nclasses
        # estimate prior
        nsamples = np.bincount(y.flatten())
        self.priors =  nsamples.astype(float) / y.shape[0]

        # estimate mus
        self.mus = (X.T @ onehot).T / nsamples.reshape(-1, 1)

        # estimate sigma
        self.sigma = (X - self.mus[y.flatten()]).T @ np.diag(self.priors[y.flatten()]) @ (X - self.mus[y.flatten()])
        
        self.invSigma = np.linalg.inv(self.sigma)  # (p, p)
    
    def predict(self, X):
        diff = X[:, None, :] - self.mus[None, :, :]   # (n, c, p)
        mdist = -0.5 * np.einsum('ncd,dd,ncd->nc', diff, self.invSigma, diff) #  n, c
        logits = mdist +  np.log(self.priors)[None]
        return np.argmax(logits, axis=1)

    def predict_proba(self, X):
        diff = X[:, None, :] - self.mus[None, :, :]   # (n, c, p)
        mdist = -0.5 * np.einsum('ncd,dd,ncd->nc', diff, self.invSigma, diff) 
        logits = mdist +  np.log(self.priors)[None] #  n, c
        # avoid overflow: np.exp(100) -> inf
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def plot_decision_boundary(self, X, y, h=0.02):
        """
        X: (n,2) 只支持二维可视化
        y: (n,) 真实标签
        h: 网格步长
        """
        if X.shape[1] != 2:
            raise ValueError("只能画二维数据的决策边界")

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=20, cmap=plt.cm.coolwarm)
        plt.title("GDA Decision Boundary")
        plt.show()


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0, class_sep=3, n_clusters_per_class=1, random_state=42)
    y = y.reshape(-1, 1)

    model = GDA()
    model.fit(X, y)
    y_hat = model.predict(X)
    print(y_hat.shape, y.shape)
    print("GDA accuracy:", accuracy_score(y, y_hat))

    plt.scatter(X[y_hat==0, 0], X[y_hat==0, 1], c='r', s=10)
    plt.scatter(X[y_hat==1, 0], X[y_hat==1, 1], c='b', s=10)
    plt.show()
    model.plot_decision_boundary(X, y.flatten())

