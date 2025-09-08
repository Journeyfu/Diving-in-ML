# https://towardsdatascience.com/a-step-by-step-implementation-of-principal-component-analysis-5520cc6cd598/
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

# step 0: generate synthetic data

mu = np.array([10, 13])
sigma = np.array([[3.5, -1.8], [-1.8, 3.5]])
org_data = rnd.multivariate_normal(mu, sigma, 1000)

# plt.scatter(org_data[:, 0], org_data[:, 1], alpha=0.5)
# plt.show()
# step 1: compute the covariance matrix

# step 2: compute the eigenvalues and eigenvectors of the covariance matrix

# step 3: sort the eigenvalues and eigenvectors

# step 4: choose the top k eigenvectors to form a new matrix

# step 5: transform the original data using the new matrix

# step 6: visualize the transformed data

# step 7: reconstruct the original data from the transformed data (optional)

# step 8: evaluate the reconstruction error (optional)

