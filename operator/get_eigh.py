
import numpy as np
import scipy


def get_eigh(A, method="Jacobi"):
    if method == "definition": # 直接调用定义, 适合小矩阵, 复杂度高, 解(A - λI)v = 0
        return get_definition_eigh(A)
    if method == "Jacobi":# 求全部特征值/特征向量, 适合对称矩阵, 复杂度低
        return get_Jacobi_eigh(A)
    elif method == "Power_iter_Rayleigh": # 求最大/最小特征值, 复杂度适中
        return get_Power_iter_Rayleigh_eigh(A)
    elif method == "QR": # 求全部特征值/特征向量, 适合一般矩阵, 复杂度高
        return get_QR_eigh(A)
    else:
        raise ValueError("Unknown method: {}".format(method))

def get_definition_eigh(A):
    # np.roots: Find the roots (soluations) of a polynomial with coefficients(from highest degree term to constant term) given in p.
    # np.poly: inverse process of np.roots, return the coefficients(from highest degree term to constant term) of the polynomial whose roots are the elements of x.
    pass


def get_Jacobi_eigh(A):
    pass
        
def get_Power_iter_Rayleigh_eigh(A):
    pass

def get_QR_eigh(A):
    pass

if __name__ == "__main__":

    # Example symmetric matrix
    A = np.array([[4, 1, 2],
                  [1, 3, 0],
                  [2, 0, 5]])

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(A)

    print("scipy Eigenvalues:", eigenvalues)
    print("scipy Eigenvectors:\n", eigenvectors)

    eigvals, eigvecs = np.linalg.eigh(A)
    print("numpy Eigenvalues:", eigvals)
    print("numpy Eigenvectors:\n", eigvecs)

    for name in ["definition", "Jacobi", "Power_iter_Rayleigh", "QR"]:
        eigvals, eigvecs = get_eigh(A, method=name)
        print(f"{name} Eigenvalues:", eigvals)
        print(f"{name} Eigenvectors:\n", eigvecs)