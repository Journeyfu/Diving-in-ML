# LVQ3 implementation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# On top of LVQ2.1, LVQ3 introduces a same-class window update when two closest prototypes are from same class with current sample.
class LVQ3:
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.01, n_epochs=100, epsilon=0.2, xi=0.1):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.prototypes = None
        self.prototype_labels = None

        self.epsilon = epsilon
        self.xi = xi
    
    def fit(self, X, y):
        self.prototypes = []
        self.prototype_labels = []

        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            idx = np.random.choice(len(X_c), self.n_prototypes_per_class, replace=False)
            self.prototypes.append(X_c[idx])
            self.prototype_labels += [c] * self.n_prototypes_per_class

        self.prototypes = np.vstack(self.prototypes)
        self.prototype_labels = np.array(self.prototype_labels)

        for epoch in range(self.n_epochs):
            eta = self.learning_rate * (1 - epoch / (self.n_epochs - 1))
            for sample, label in zip(X, y):
                dist = np.linalg.norm(self.prototypes - sample[None], axis=-1)
                closest_idx, dist_list, label_list = self.choose_two_closest_prototypes(dist)
                
                if self.decision_update(label, dist_list, label_list):

                    if label_list[0] == label_list[1] == label:
                        self.prototypes[closest_idx[0]] += eta * self.xi * (sample - self.prototypes[closest_idx[0]])
                        self.prototypes[closest_idx[1]] += eta * self.xi * (sample - self.prototypes[closest_idx[1]])
                    else:
                        for i in range(2):
                            if self.prototype_labels[closest_idx[i]] == label:
                                self.prototypes[closest_idx[i]] += eta * (sample - self.prototypes[closest_idx[i]])
                            else:
                                self.prototypes[closest_idx[i]] -= eta * (sample - self.prototypes[closest_idx[i]])
        
    def choose_two_closest_prototypes(self, dist):
        sorted_idx = np.argsort(dist)
        closest_idx = [sorted_idx[0], sorted_idx[1]]
        dist_list = [dist[sorted_idx[0]], dist[sorted_idx[1]]]
        label_list = [self.prototype_labels[sorted_idx[0]], self.prototype_labels[sorted_idx[1]]]
        return closest_idx, dist_list, label_list

    def decision_update(self, label, dist, label_list):
        if label not in label_list: 
            return False
        elif label_list[0] == label_list[1] == label: # all from same class w/o window condition
            return True
        elif min(dist) / max(dist) > (1 - self.epsilon) / (1 + self.epsilon): # one is from same class, other is not, with window condition
            return True
        else:
            return False
    
    def predict(self, X):
        dist = np.linalg.norm(self.prototypes[None]-X[:, None], axis=-1)
        return self.prototype_labels[np.argmin(dist, axis=-1)]


if __name__ == "__main__":
    true_n_clusters = 10
    X, y_true = make_blobs(
        n_samples=2000, centers=true_n_clusters, cluster_std=2, random_state=42
    )
    indices = np.random.permutation(len(X))
    X, y_true = X[indices], y_true[indices]

    model = LVQ3(n_prototypes_per_class=3, learning_rate=0.01, n_epochs=20)
    model.fit(X, y_true)
    y_pred = model.predict(X)

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("GT")
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, alpha=0.5, cmap="jet")
    plt.scatter(model.prototypes[:, 0], model.prototypes[:, 1], color="k", s=80, marker='x')
    plt.scatter(model.prototypes[:, 0], model.prototypes[:, 1], c=model.prototype_labels, s=50, marker='x', cmap="jet")
    plt.title("LVQ3 Clustering")
    plt.show()