from data import get_2d_noise_linear_data, get_log_curve_data
import numpy as np
import matplotlib.pyplot as plt

class MSE_Solver:
    def __init__(self,):
        self.W = None
        self.lr, self.num_iters = 0.1, 2000

    def predict(self, X): # X: n x d
        return X @ self.W
        
    def fit(self, X, y):
        X_expand = np.hstack((X, np.ones((X.shape[0], 1))))

        self.W = np.linalg.inv(X_expand.T @ X_expand) @ X_expand.T @ y

        y_hat = X_expand @ self.W
        plt.subplot(1, 3, 1)
        plt.scatter(X[:, :1], y, s=10)
        plt.plot(X[:, :1], y_hat, c="r")
        plt.title("Closed form solution")

        self.W = None
        # only support linear equation input , such as AW = B
        self.W = np.linalg.solve(X_expand.T @ X_expand, X_expand.T @ y)
        y_hat = X_expand @ self.W

        plt.subplot(1, 3, 2)
        plt.scatter(X[:, :1], y, s=10)
        plt.plot(X[:, :1], y_hat, c="r")
        plt.title("np.linlg.solve solution")


        self.W = None
        plt.subplot(1, 3, 3)
        self.W = np.random.randn(X_expand.shape[1], 1) * 0.01
        X_norm = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
        X_norm_expand = np.hstack((X_norm, np.ones((X.shape[0], 1))))
        # the scale of y doesn't change

        for it in range(self.num_iters):
            y_hat = (X_norm_expand @ self.W)
            grad = 2 * X_norm_expand.T @ (y_hat - y) / y.shape[0] 
            self.W = self.W - self.lr * np.clip(grad, -1e5, 1e5)

            if it % 15 == 0:
                y_hat = X_norm_expand @ self.W
                plt.plot(X[:, :1], y_hat, c="r", alpha=min(float(it) / self.num_iters, 0.1))

        y_hat = X_norm_expand @ self.W 
        
        plt.scatter(X[:, :1], y, s=10)
        plt.plot(X[:, :1], y_hat, c="r")
        plt.title("Iterative solution")
        plt.subplots_adjust(wspace=1)
        plt.show()


if __name__ == "__main__":
    # X, y = get_2d_noise_linear_data(100, outlier_fraction=0.01, seed=42) 

    # mse = MSE_Solver()
    # mse.fit(X, y)

    X, y = get_log_curve_data(100, seed=42) 

    X = np.hstack((X, X**2))

    mse = MSE_Solver()
    mse.fit(X, y)
    
        

