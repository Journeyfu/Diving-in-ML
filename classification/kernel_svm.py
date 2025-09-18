import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from tqdm import tqdm
import matplotlib.pyplot as plt
from classification.linear_svm import LinearSVM


class KernelSVM: # better implementation than linear svm
    def __init__(self):
        self.lam = None
        self.w = None

        self.b = 0.
        self.tol = 1e-4
        self.C = 1.0

    def select_two_lambda(self, X, y, gx, E):
        # select one that violates KKT condition
        candidates = []
        for si in range(X.shape[0]):
            gx_si = gx[si]
            # if 0 < self.lam[si] < self.C and not (1 - self.tol <= y[si] * gx_si <= 1 + self.tol):
            #     candidates.append(si)
            # elif self.lam[si] == 0 and not (y[si] * gx_si >= 1 - self.tol):
            #     candidates.append(si)
            # elif self.lam[si] == self.C and not (y[si] * gx_si <= 1 + self.tol):
            #     candidates.append(si)
            
            if (y[si] * gx_si < 1 - 1e-5 and self.lam[si] < self.C) or \
               (y[si] * gx_si > 1 + 1e-5 and self.lam[si] > 0):
                candidates.append(si)

        
        if len(candidates) == 0:
            return -1, -1, True

        l1 = np.random.choice(candidates)
        
        # select the second one that has maximum |E1 - E2|
        l2 = np.argmax(np.abs(E[l1]-E))
        return l1, l2, False
    
    def get_lam2_feasible_bounds(self, y, l1, l2):
        if y[l1] != y[l2]:
            L = max(0, self.lam[l2] - self.lam[l1])
            H = min(self.C, self.C + self.lam[l2] - self.lam[l1])
        else:
            L = max(0, self.lam[l1] + self.lam[l2] - self.C)
            H = min(self.C, self.lam[l1] + self.lam[l2])
        return L, H

    def fit(self, X, y, max_iter=1000, C=1.0):
        self.lam = np.zeros(X.shape[0], dtype=np.float32)
        self.K = self.kernel(X, X)
        self.C = C

        for it in tqdm(range(max_iter)):
            gx = np.dot(self.lam * y, self.K) + self.b
            E = gx - y
            l1, l2, flag = self.select_two_lambda(X, y, gx, E)
            if flag == True:
                print(f"converged at iteration {it}")
                break
            E1, E2 = E[l1], E[l2]
            k11 = self.K[l1, l1]
            k22 = self.K[l2, l2]
            k12 = self.K[l1, l2]

            l2_L, l2_H = self.get_lam2_feasible_bounds(y, l1, l2)
            if l2_L == l2_H:
                continue

            eta = k11 + k22 - 2 * k12
            if eta <= 0:
                continue

            new_l2 = self.lam[l2] + y[l2] * (E1 - E2) / eta
            new_l2 = np.clip(new_l2, l2_L, l2_H)

            if abs(new_l2 - self.lam[l2]) < self.tol:
                continue

            new_l1 = self.lam[l1] + y[l1] * y[l2] * (self.lam[l2] - new_l2)
            
            new_b1 = self.b - E1 - y[l1] * k11 * (new_l1 - self.lam[l1]) - y[l2] * k12 * (new_l2 - self.lam[l2])
            new_b2 = self.b - E2 - y[l1] * k12 * (new_l1 - self.lam[l1]) - y[l2] * k22 * (new_l2 - self.lam[l2])

            if 0 < new_l1 < self.C:
                self.b = new_b1
            elif 0 < new_l2 < self.C:
                self.b = new_b2
            else:
                self.b = (new_b1 + new_b2) / 2  

            self.lam[l1] = new_l1
            self.lam[l2] = new_l2
        
        mask = self.lam > 1e-6
        # mask = (self.lam > 0) & (self.lam < self.C)
        self.support_x = X[mask]
        self.support_y = y[mask]
        self.support_lam = self.lam[mask]
        if len(self.support_x) == 0:
            print("no support vector found")
        else:
            print(f"number of support vectors: {len(self.support_x)}")

    def rbf_kernel(self, x1, x2, gamma=None): # rbf kernel
        if gamma is None:
            gamma = 1.0 / x1.shape[1]
        diff = x1[:, None] - x2[None, :]
        rbf_dist = np.exp(-gamma * np.sum(diff**2, axis=-1))
        return rbf_dist

    def polynomial_kernel(self, x1, x2, degree=2, alpha=1.0, coef0=0.0):  # polynomial kernel
        return (alpha * np.dot(x1, x2.T) + coef0) ** degree

    def kernel(self, x1, x2):
        # return self.polynomial_kernel(x1, x2)
        return self.rbf_kernel(x1, x2, gamma=2.0)
    
    def predict(self, X):
        K = self.kernel(self.support_x, X)
        y_hat = np.sum(self.support_lam[:, None] * self.support_y[:, None] * K, axis=0) + self.b
        return np.sign(y_hat)

    def plot_decision_boundary(self, X, y):
        plt.figure(figsize=(8,6))
        plt.scatter(X[y==-1, 0], X[y==-1, 1], c='r', s=20, label='-1')
        plt.scatter(X[y==1, 0], X[y==1, 1], c='b', s=20, label='1')
        plt.scatter(self.support_x[:,0], self.support_x[:,1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

        x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
        y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
        xx = np.linspace(x_min, x_max, 200)
        yy = np.linspace(y_min, y_max, 200)
        XX, YY = np.meshgrid(xx, yy)

        grid_points = np.c_[XX.ravel(), YY.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(XX.shape)

        plt.contour(XX, YY, Z, levels=[-1,0,1], linestyles=['--','-','--'], colors='k')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    X, y = make_moons(
        n_samples=1000, noise=0.1, random_state=0)
    y = np.where(y==0, -1, 1)  # labels should be in {-1, 1}
    
    model = KernelSVM()
    model.fit(X, y, max_iter=100000, C=10.0)
    y_hat = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_hat))

    model.plot_decision_boundary(X, y)

    from sklearn.svm import SVC
    sklearn_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    sklearn_model.fit(X, y)
    y_sklearn = sklearn_model.predict(X)
    print("Sklearn SVM Accuracy:", accuracy_score(y, y_sklearn))
