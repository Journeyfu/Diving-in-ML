import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

class LogisticRegression():
    def __init__(self):
        self.W = None
        self.lr, self.num_iters = 0.1, 2000
    
    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))
    
    def predict(self, X, is_expand=True): # X: n x d
        if not is_expand:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        return self.sigmoid(X @ self.W)

    def fit(self, X, y):
        X_expand = np.hstack((X, np.ones((X.shape[0], 1))))
        self.W = np.random.randn(X_expand.shape[1], 1)

        for it in range(self.num_iters):
            z = self.predict(X_expand) # N x 1
            # non-vectorized version
            # grad = (y * (1 - z) * X_expand   + (1 - y) * (-X_expand) * z)
            # grad = grad.sum(axis=0)[:, None]

            # vectorized version
            grad = X_expand.T @ (z - y)
            self.W = self.W - self.lr * grad

        y_hat = (self.predict(X_expand) > 0.5).astype(int)

        y_hat = y_hat.reshape(-1)
        y = y.reshape(-1)
        plt.subplot(1, 2, 1)
        plt.scatter(X[y==0, 0], X[y==0, 1], c='r', s=10)
        plt.scatter(X[y==1, 0], X[y==1, 1], c='b', s=10)
        plt.title("GT")

        plt.subplot(1, 2, 2)
        plt.scatter(X[y_hat==0, 0], X[y_hat==0, 1], c='r', s=10)
        plt.scatter(X[y_hat==1, 0], X[y_hat==1, 1], c='b', s=10)
        plt.title("logitstic regression")
        plt.subplots_adjust(wspace=1)
        plt.show()

    def plot_decision_boundary(self, X, y, steps=100):
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_expand = np.hstack((grid, np.ones((grid.shape[0],1))))  # 加 bias
    
        probs = self.predict(grid_expand).reshape(xx.shape)
    
        plt.contourf(xx, yy, probs, levels=[0,0.5,1], alpha=0.2, colors=['red','blue'])
        plt.scatter(X[y[:,0]==0,0], X[y[:,0]==0,1], c='r', s=10, label='Class 0 gt')
        plt.scatter(X[y[:,0]==1,0], X[y[:,0]==1,1], c='b', s=10, label='Class 1 gt')
        plt.title("Logistic Regression with Decision Boundary")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0, class_sep=3, n_clusters_per_class=1, random_state=42)
    y = y.reshape(-1, 1)

    model = LogisticRegression()
    model.fit(X, y)
    model.plot_decision_boundary(X, y)
    acc = accuracy_score(y, (model.predict(X, is_expand=False) > 0.5).astype(int))
    print(f"Accuracy: {acc*100:.2f}%")

