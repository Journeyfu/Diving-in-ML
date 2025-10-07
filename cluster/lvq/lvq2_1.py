# learning Vector Quantization (LVQ) clustering v2.1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from cluster.lvq.lvq2 import LVQ2

# On top of LVQ2, LVQ2.1 updates the prototypes only when there is at least one positive prototype
class LVQ2_1(LVQ2):
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.01, n_epochs=100, epsilon=0.2):
        super().__init__(n_prototypes_per_class, learning_rate, n_epochs, epsilon)
        
    def decision_update(self, label, idx, dist):
        label_list = [self.prototype_labels[i] for i in idx]
        if label not in label_list:
            return False

        if min(dist) / max(dist) > (1 - self.epsilon) / (1 + self.epsilon):
            return True
        else:
            return False
    

if __name__ == "__main__":
    
    true_n_clusters = 10
    X, y_true = make_blobs(
        n_samples=2000, centers=true_n_clusters, cluster_std=2, random_state=42
    )

    model = LVQ2_1(n_prototypes_per_class=3, learning_rate=0.01, n_epochs=20, epsilon=0.2)
    model.fit(X, y_true)
    y_pred = model.predict(X)

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="jet")
    plt.title("GT")
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, alpha=0.5, cmap="jet")
    plt.scatter(model.prototypes[:, 0], model.prototypes[:, 1], c="k", s=80, marker='x', cmap="jet")
    plt.scatter(model.prototypes[:, 0], model.prototypes[:, 1], c=model.prototype_labels, s=50, marker='x', cmap="jet")
    plt.title("LVQ2.1 Clustering")
    plt.show()

