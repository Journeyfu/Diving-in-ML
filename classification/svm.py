import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from tqdm import tqdm
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self):
        self.lam = None

        self.b = 0.
        self.tol = 1e-4
        self.C = 1.0
        self.max_iter = 1000

    def select_two_lambda(self, X, y):
        # select one that violates KKT condition
        candidates = []
        for si in range(X.shape[0]):
            gx = self.get_gx(X, y, si)
            if 0 < self.lam[si] < self.C and not (1 - self.tol <= y[si] * gx <= 1 + self.tol):
                candidates.append(si)
            elif self.lam[si] == 0 and not (y[si] * gx >= 1 - self.tol):
                candidates.append(si)
            elif self.lam[si] == self.C and not (y[si] * gx <= 1 + self.tol):
                candidates.append(si)
        
        if len(candidates) == 0:
            return -1, -1

        l1 = np.random.choice(candidates)
        
        # select the second one that has maximum |E1 - E2|
        e1 = self.error(X, y, l1)
        max_err, l2 = -1, -1
        for si in range(X.shape[0]):
            if si == l1:
                continue
            e2 = self.error(X, y, si)
            if abs(e1 - e2) > max_err:
                max_err = abs(e1 - e2)
                l2 = si

        return l1, l2
    
    def get_gx(self, X, y, li): # y: (n, )
        return np.dot(self.lam * y, self.linear_kernel(X, X[li:li+1])) + self.b
    
    def error(self, X, y, li):
        return self.get_gx(X, y, li) - y[li]

    def linear_kernel(self, x1, x2): 
        return x1 @ x2.T

    def get_lam2_feasible_bounds(self, y, l1, l2):
        if y[l1] != y[l2]:
            L = max(0, self.lam[l2] - self.lam[l1])
            H = min(self.C, self.C + self.lam[l2] - self.lam[l1])
        else:
            L = max(0, self.lam[l1] + self.lam[l2] - self.C)
            H = min(self.C, self.lam[l1] + self.lam[l2])
        return L, H
    
    def fit(self, X, y):
        self.lam = np.zeros(X.shape[0], dtype=np.float32)

        for it in tqdm(range(self.max_iter)):
            l1, l2 = self.select_two_lambda(X, y)
            # print(f"it {it}, 11 {l1}, l2 {l2}")
            if l1 == -1 and l2 == -1:
                print(f"converged at iteration {it}")
                break

            e1 = self.error(X, y, l1)
            e2 = self.error(X, y, l2)
            k11 = self.linear_kernel(X[l1:l1+1], X[l1:l1+1]).item()
            k22 = self.linear_kernel(X[l2:l2+1], X[l2:l2+1]).item()
            k12 = self.linear_kernel(X[l1:l1+1], X[l2:l2+1]).item()
            l2_L, l2_H = self.get_lam2_feasible_bounds(y, l1, l2)
            if l2_L == l2_H:
                continue
            eta = k11 + k22 - 2 * k12
            if eta <= 0:
                continue

            new_l2 = self.lam[l2] + y[l2] * (e1 - e2) / eta
            new_l2 = np.clip(new_l2, l2_L, l2_H)

            if abs(new_l2 - self.lam[l2]) < self.tol:
                continue

            new_l1 = self.lam[l1] + y[l1] * y[l2] * (self.lam[l2] - new_l2)
            
            new_b1 = self.b - e1 - y[l1] * k11 * (new_l1 - self.lam[l1]) - y[l2] * k12 * (new_l2 - self.lam[l2])
            new_b2 = self.b - e2 - y[l1] * k12 * (new_l1 - self.lam[l1]) - y[l2] * k22 * (new_l2 - self.lam[l2])

            if 0 < new_l1 < self.C:
                self.b = new_b1
            elif 0 < new_l2 < self.C:
                self.b = new_b2
            else:
                self.b = (new_b1 + new_b2) / 2  

            self.lam[l1] = new_l1
            self.lam[l2] = new_l2
        
        self.support_x = X[self.lam > 0]
        self.support_y = y[self.lam > 0]
        self.support_lam = self.lam[self.lam > 0]
        if len(self.support_x) == 0:
            print("no support vector found")
        
    def predict(self, X):
        y_hat =  np.sign(np.dot(self.support_lam * self.support_y, self.linear_kernel(self.support_x, X)) + self.b)
        return y_hat.ravel()

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0, class_sep=3, flip_y=0, n_clusters_per_class=1, random_state=0)
    y = np.where(y==0, -1, 1)  # labels should be in {-1, 1}
    
    print(X.shape, y.shape) # (1000, 2) (1000,)
    model = LinearSVM()
    model.fit(X, y)
    y_hat = model.predict(X)
    print("Accuracy:", accuracy_score(y, y_hat))

    #plt.scatter(X[y==-1, 0], X[y==-1, 1], c='r', s=10)
    #plt.scatter(X[y==1, 0], X[y==1, 1], c='b', s=10)
    plt.scatter(X[y_hat==-1, 0], X[y_hat==-1, 1], c='r', s=10)
    plt.scatter(X[y_hat==1, 0], X[y_hat==1, 1], c='b', s=10)
    plt.show()