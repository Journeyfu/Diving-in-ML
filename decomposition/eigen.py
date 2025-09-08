import numpy as np
import sympy as sp

def my_eigen(matrix, method="jacobi"):
    if method == "jacobi":
        return solve_by_jacobi(matrix)
    elif method == "power_iteration":
        return solve_by_power_iteration(matrix)
    elif method == "qr_algorithm":
        return solve_by_qr_algorithm(matrix)
    elif method == "definition_sympy":
        return solve_by_definition_sympy(matrix)
    elif method == "definition_iteration":
        return solve_by_definition_iter(matrix)
    else:
        raise Exception("Unknown method")

def solve_by_definition_sympy(matrix):
    # let det(A - \lambda I) = 0
    pass

def solve_by_definition_iter(matrix):
    # let det(A - \lambda I) = 0
    pass

def solve_by_jacobi(matrix):
    pass

def solve_by_power_iteration(matrix):
    pass

def solve_by_qr_algorithm(matrix):
    pass
    

if __name__ == "__main__":
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.linalg.eig(A)

    # my_eigen(A, method="jacobi")
    # my_eigen(A, method="power_iteration")
    # my_eigen(A, method="qr_algorithm")
    my_eigen(A, method="definition")
