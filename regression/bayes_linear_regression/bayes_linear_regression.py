import numpy as np
from data import get_2d_noise_linear_data, get_log_curve_data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge


class BayesLinearRegression:
    def __init__(self,):
        self.noise_var = None # TODO: use inverse gamma prior for noise variance
        self.W_prior_cov = None
        self.mu = None
        self.sigma = None

    def init(self, dim):
        self.noise_var = 1.0
        self.W_prior_cov = np.eye(dim) * 1.

    def fit(self, X, y): # infer weight distribution
        self.init(dim=X.shape[1])
        precision_matrix = self.noise_var**(-1) * X.T @ X + np.linalg.inv(self.W_prior_cov)
        self.sigma = np.linalg.inv(precision_matrix)
        self.mu = self.noise_var**(-1) * self.sigma @ X.T @ y

    def predict(self, X): # predict probabilistic distribution of y
        y_mu = X @ self.mu
        y_var = np.sum(X @ self.sigma * X, axis=-1) + self.noise_var

        return y_mu, y_var


if __name__ == "__main__":
    X, y = get_2d_noise_linear_data(100, seed=42)
    # X, y = get_log_curve_data(100, seed=42)  # 非线性，可能需要多项式特征

    X = np.hstack((X, np.ones((X.shape[0], 1))))  # add bias

    
    y = np.asarray(y, dtype=float).ravel()

    blr = BayesLinearRegression()
    blr.fit(X, y)
    y_mu, y_var = blr.predict(X)
    y_std = np.sqrt(y_var)

    order = np.argsort(X[:, 0])
    x_sorted = X[order, 0]          # 一维
    y_sorted = y_mu[order]          # 一维
    std_sorted = y_std[order]       # 一维

    plt.scatter(X[:, 0], y, s=10, label="data")
    plt.plot(x_sorted, y_sorted, c="r", label="BLR mean")
    plt.fill_between(
        x_sorted, y_sorted - 2 * std_sorted, y_sorted + 2 * std_sorted,
        alpha=0.2, label="~95% CI"
    )
    plt.legend()
    plt.title("Bayesian Linear Regression")
    plt.show()