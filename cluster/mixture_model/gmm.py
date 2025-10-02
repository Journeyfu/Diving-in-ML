import numpy as np
from data import get_2d_mixture_data
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


class MyGaussianMixture:
    def __init__(self, n_components=3, random_state=None):
        self.n_components = n_components
        self.max_iter = 100
        self.tol = 1e-4
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihoods_ = []
        self.rng = np.random.default_rng(random_state)

    def fit(self, X):
        self.init(X)
        for n_it in range(self.max_iter):
            responsibilities = self.E_step(X)
            self.M_step(X, responsibilities)

    def init(self, X): # X: (n_samples, n_features)
        dim = X.shape[-1]
        means = self.rng.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=self.n_components)
        covariances = np.array([np.eye(dim)] * self.n_components)
        weights = np.ones(self.n_components) / self.n_components

        self.means_ = means # shape (n_components, n_features)
        self.covariances_ = covariances # shape (n_components, n_features, n_features)
        self.weights_ = weights # shape (n_components,)
    
    def E_step(self, X):
        dim = X.shape[-1]
        diff = X[:, None] - self.means_[None]  # shape (n_samples, n_components, n_features)
        inv_cov = np.linalg.inv(self.covariances_)  # shape (n_components, n_features, n_features)
        norm_term = (1. / ((2* np.pi)**(dim/2) * np.sqrt(np.linalg.det(self.covariances_))))[None] # shape (1, n_components) 
        mahalanobis = np.einsum('nkd, kde, nke->nk', diff, inv_cov, diff) # shae (n_samples, n_components)
        probabilities = norm_term*np.exp(-0.5 * mahalanobis)
        responsibilities = self.weights_[None] * probabilities
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True) # shape (n_samples, n_components)
        return responsibilities

    def M_step(self, X, responsibilities): 
        # update weights
        self.weights_ = responsibilities.sum(axis=0) / X.shape[0]  # shape (n_components,)
        # update means 
        self.means_ = (responsibilities[:, :, None] * X[:, None, :]).sum(axis=0) / responsibilities[:, :, None].sum(axis=0)
        # update covariances
        diff = X[:, None] - self.means_[None]
        self.covariances_ = np.einsum('nk, nkd, nkf -> kdf', responsibilities, diff, diff) / responsibilities.sum(axis=0)[:, None, None]

    def predict(self, X):
        responsibilities = self.E_step(X) # shape (n_samples, n_components)
        pred = responsibilities.argmax(axis=-1) # shape (n_samples,)
        return pred



if __name__ == "__main__":
    seed = 0
    num_points_per_cluster = 300
    num_components = 5

    # Generate synthetic data
    X, true_labels = get_2d_mixture_data(num_points_per_cluster, num_components, seed=seed)
    # X = X / np.linalg.norm(X, axis=-1, keepdims=True)

    # Plot the results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    for i in np.unique(true_labels):
        plt.scatter(X[true_labels == i, 0], X[true_labels == i, 1], s=10, label=f'True Cluster {i}')
    plt.title("True Clusters")
    plt.legend(loc='upper right')
    plt.axis("equal")

    my_gmm = MyGaussianMixture(n_components=num_components, random_state=seed)
    my_gmm.fit(X)
    my_labels = my_gmm.predict(X)
    my_means = my_gmm.means_

    plt.subplot(1, 3, 2)
    for i in np.unique(my_labels):
        plt.scatter(X[my_labels == i, 0], X[my_labels == i, 1], s=10, label=f'Predicted Cluster {i}')
    plt.scatter(my_means[:, 0], my_means[:, 1], c='red', s=10, marker='X', label='GMM Means')
    plt.title("My GMM Predicted Clusters")
    plt.legend(loc='upper right')
    plt.axis("equal")

    # Fit a Gaussian Mixture Model using scikit-learn
    sk_gmm = GaussianMixture(n_components=num_components, random_state=seed)
    sk_gmm.fit(X)
    sk_labels = sk_gmm.predict(X)
    sk_means = sk_gmm.means_

    plt.subplot(1, 3, 3)
    for i in np.unique(sk_labels):
        plt.scatter(X[sk_labels == i, 0], X[sk_labels == i, 1], s=10, label=f'Predicted Cluster {i}')
    plt.scatter(sk_means[:, 0], sk_means[:, 1], c='red', s=10, marker='X', label='GMM Means')
    plt.title("sk-learn GMM Predicted Clusters")
    plt.legend(loc='upper right')
    plt.axis("equal")

    plt.show()