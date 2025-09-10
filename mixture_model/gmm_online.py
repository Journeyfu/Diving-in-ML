import numpy as np
from data import get_2d_mixture_data
import matplotlib.pyplot as plt

class OnlineGaussianMixture:
    def __init__(self, n_components=3, random_state=None):
        self.n_components = n_components
        self.max_iter = 100
        self.tol = 1e-4
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihoods_ = []
        self.rng = np.random.default_rng(random_state)


    def E_step(self, X):
        pass

    def M_step(self, X, responsibilities):
        pass

    def predict(self, X):
        pass


if __name__ == "__main__":
    seed = 0
    num_points_per_cluster = 300
    num_components = 5

    # Generate synthetic data
    X, true_labels = get_2d_mixture_data(num_points_per_cluster, num_components, seed=seed)