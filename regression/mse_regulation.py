from data import get_3d_noise_linear_data, get_log_curve_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# TODO: add sklearn version for comparison
# understand L1 reg detail
from sklearn.linear_model import LinearRegression, Ridge, Lasso


class MSE_Solver:
    def __init__(self,):
        self.W = None
        # let lr to 0.1, num_iters set to 20 is enough
        # self.lr, self.num_iters = 0.1, 20
        self.lr, self.num_iters = 0.01, 150


    def soft_threshold(self, rho, lam):
        return np.sign(rho) * max(abs(rho) - lam, 0)
        
    def fit_3d_data(self, X, y):
        X_expand = np.hstack((X, np.ones((X.shape[0], 1))))

        self.W = np.linalg.inv(X_expand.T @ X_expand ) @ X_expand.T @ y
        y_hat = X_expand @ self.W

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(231, projection='3d')
        ax.scatter(X[:,0], X[:,1], y, s=50)
        ax.plot_trisurf(X[:,0], X[:,1], y_hat.flatten(), color='red', alpha=0.5)    
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        ax.set_title("Closed form solution")

        ## L2 regularization + standardization

        ax = fig.add_subplot(232, projection='3d')
        ax.scatter(X[:,0], X[:,1], y, s=50)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        ax.set_title("L2 Regularization")

        X_norm = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
        X_expand = np.hstack((X_norm, np.ones((X.shape[0], 1))))

        c_list = ['b', 'g', 'r', 'c', 'm']
        handles = []
        for idx, l2_lambda in enumerate([1, 10, 100, 10000]):
            L2_regularization = l2_lambda * np.eye(X_expand.shape[1])
            L2_regularization[-1, -1] = 0  # Do not regularize the bias term

            self.W = np.linalg.inv(X_expand.T @ X_expand + L2_regularization ) @ X_expand.T @ y
            y_hat = X_expand @ self.W

            ax.plot_trisurf(X[:,0], X[:,1], y_hat.flatten(),
                            color=c_list[idx], alpha=0.5)

            handles.append(Patch(facecolor=c_list[idx], label=f"L2 λ={l2_lambda}", alpha=0.5))

        ax.legend(handles=handles, loc="lower right", fontsize=8)

        ## L1 regularization + standardization (Coordinate Descent)
        self.W = np.random.randn(X_expand.shape[1], 1) * 0.01
        X_norm = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
        X_expand = np.hstack((X_norm, np.ones((X.shape[0], 1))))
        # the scale of y doesn't change

        L1_lambda = 1
        # my version
        for it in range(self.num_iters):
            for j in range(self.W.shape[0]):
                r_ij = y - (X_expand @ self.W) + X_expand[:, j:j+1] * self.W[j:j+1] # Nx1
                
                norm_j = (X_expand[:, j:j+1]**2).sum(axis=0)

                z_j = (X_expand[:, j:j+1].T @ r_ij).sum()
                # z_j = (r_ij * X_expand[:, j:j+1]).sum() # same form

                self.W[j] =  self.soft_threshold(z_j, L1_lambda / 2.)/ norm_j

        y_hat = X_expand @ self.W 

        ax = fig.add_subplot(233, projection='3d')
        ax.scatter(X[:,0], X[:,1], y, s=50)
        ax.plot_trisurf(X[:,0], X[:,1], y_hat.flatten(), color=c_list[idx], alpha=0.5)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        ax.set_title("L1 Regularization")


        # sklearn version MSE
        ax = fig.add_subplot(234, projection='3d')
        md = LinearRegression().fit(X, y)
        y_hat = md.predict(X)

        ax.scatter(X[:,0], X[:,1], y, s=50)
        ax.plot_trisurf(X[:,0], X[:,1], y_hat.flatten(), color=c_list[idx], alpha=0.5)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        ax.set_title("sklearn MSE")


        # sklearn version MSE + L2
        ax = fig.add_subplot(235, projection='3d')
        md = Ridge().fit(X, y)
        y_hat = md.predict(X)
        ax.scatter(X[:,0], X[:,1], y, s=50)
        ax.plot_trisurf(X[:,0], X[:,1], y_hat.flatten(), color=c_list[idx], alpha=0.5)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        ax.set_title("sklean L2 Regularization")

        # sklearn version MSE + L1
        ax = fig.add_subplot(236, projection='3d')
        md = Lasso().fit(X, y)
        y_hat = md.predict(X)
        ax.scatter(X[:,0], X[:,1], y, s=50)
        ax.plot_trisurf(X[:,0], X[:,1], y_hat.flatten(), color=c_list[idx], alpha=0.5)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        ax.set_title("sklearn L1 Regularization")
        

        # plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    X, y = get_3d_noise_linear_data(100, outlier_fraction=0.3,  seed=42) 

    mse = MSE_Solver()
    mse.fit_3d_data(X, y)

    


 
    
        

