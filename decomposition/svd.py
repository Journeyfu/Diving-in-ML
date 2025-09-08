import numpy as np
from numpy.linalg import svd

def my_svd(matrix):
    """
    Perform Singular Value Decomposition on the given matrix.

    Parameters:
    matrix (np.ndarray): The input matrix to decompose.

    Returns:
    U (np.ndarray): Left singular vectors.
    S (np.ndarray): Singular values.
    VT (np.ndarray): Right
    """
    pass

if __name__ == "__main__":
    # Create a sample matrix
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Perform Singular Value Decomposition
    U, S, VT = svd(A)

    # Print the results
    print("Original Matrix A:")
    print(A)
    print("\nLeft Singular Vectors (U):")
    print(U)
    print("\nSingular Values (S):")
    print(S)
    print("\nRight Singular Vectors (VT):")
    print(VT)

    # Reconstruct the original matrix
    Sigma = np.zeros((U.shape[0], VT.shape[0]))
    np.fill_diagonal(Sigma, S)
    A_reconstructed = np.dot(U, np.dot(Sigma, VT))

    print("\nReconstructed Matrix A from SVD:")
    print(A_reconstructed)
