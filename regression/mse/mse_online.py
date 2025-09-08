from data import get_2d_noise_linear_data, get_log_curve_data
import matplotlib.pyplot as plt
import numpy as np

class MSE_Online:
    def __init__(self, ):
        self.W = None
    
    def predict(self, X): # X: n x d
        return X @ self.W
    
    def LSE_fit(self, X, y, lr=0.01):
        num_samples = X.shape[0]
        idx = np.random.permutation(X.shape[0])
        X_shuffled, y_shuffled = X[idx], y[idx]
        X_expand = np.hstack((X_shuffled, np.ones((X_shuffled.shape[0], 1))))

        self.W = np.random.randn(X_expand.shape[1], 1) * 0.01

        for i in range(num_samples):
            x_i = X_expand[i:i+1].T # d x 1
            y_i = y_shuffled[i:i+1]

            error = y_i - x_i.T @ self.W

            self.W = self.W + lr * x_i @ error
        
        X_expand_raw = np.hstack((X, np.ones((X_shuffled.shape[0], 1))))
        y_hat = X_expand_raw @ self.W

        plt.scatter(X[:, :1], y, s=10)
        plt.plot(X[:, :1], y_hat, c="r")
        plt.title("LSE online solution")
        plt.show()


    def RLS_fit(self, X, y):
        num_samples = X.shape[0]
        idx = np.random.permutation(X.shape[0])
        X_shuffled, y_shuffled = X[idx], y[idx]
        X_expand = np.hstack((X_shuffled, np.ones((X_shuffled.shape[0], 1))))
        num_dim = X_expand.shape[1]
        
        self.W = np.random.randn(X_expand.shape[1], 1) * 0.01

        vlambda = 0.99
        inv_vlambda = 1./vlambda
        scale = 1.
        inv_R_i = np.eye(num_dim)  * scale

        for i in range(num_samples):

            x_i = X_expand[i:i+1].T  # d x 1
            y_i = y_shuffled[i:i+1]

            err = y_i - x_i.T @ self.W
            k_n = (inv_R_i @ x_i ) / (vlambda + x_i.T @ inv_R_i @ x_i)
            self.W = self.W + k_n * err
            # inv_R_i = inv_vlambda * inv_R_i @ (np.eye( num_dim ) - k_n @ x_i.T)
            inv_R_i = inv_vlambda * (inv_R_i - k_n @ x_i.T @ inv_R_i) # recommend

        X_expand_raw = np.hstack((X, np.ones((X_shuffled.shape[0], 1))))
        y_hat = X_expand_raw @ self.W

        plt.scatter(X[:, :1], y, s=10)
        plt.plot(X[:, :1], y_hat, c="r")
        plt.title("RLS online solution")
        plt.show()



if __name__ == "__main__":
    X, y = get_log_curve_data(100, seed=42) 

    mse = MSE_Online()
    mse.LSE_fit(X, y, lr=0.01) # doest work well on polynomial features

    mse.RLS_fit(X, y)

    poly_X = np.hstack((X, X**2))
    mse.RLS_fit(poly_X, y)