import numpy as np

WIDTH = HEIGHT = 4
np.random.seed(0)
A = np.round(np.random.uniform(0, 10, (HEIGHT, WIDTH)), 1)
b = np.round(np.random.uniform(0, 10, (HEIGHT, 1)), 1)


def print_matrix(matrix, title):
    print(title)
    for line in matrix:
        print('     '.join(map(str, line)))
    print()


def lu_decomposition(matrix):
    # Get the number of rows
    n = matrix.shape[0]

    tmp_L = np.eye(n, dtype=np.double)
    tmp_U = matrix.copy()
    for i in range(n):
        # get factors of each individual row in tmp_U
        factors = tmp_U[i + 1:, i] / tmp_U[i, i]
        # assign factors to corresponding elements in tmp_L
        tmp_L[i + 1:, i] = factors
        # subtract factor * tmp_U[i] from corresponding element in
        # tmp_U, which is equivalent of doing row subtraction in linalg
        tmp_U[i + 1:] -= factors[:, np.newaxis] * tmp_U[i]
    return tmp_L, tmp_U


def forward_substitution(tmp_l, tmp_b):
    # get size of L matrix
    n = tmp_l.shape[0]

    # initialize empty array with size b
    tmp_y = np.zeros_like(tmp_b, dtype=np.double)

    # computing tmp_y values
    tmp_y[0] = tmp_b[0] / tmp_l[0, 0]
    for i in range(1, n):
        tmp_y[i] = (tmp_b[i] - np.dot(tmp_l[i, :i], tmp_y[:i])) / tmp_l[i, i]
    return tmp_y


def backward_substitution(tmp_u, tmp_y):
    # get size of U matrix
    n = tmp_u.shape[0]

    # initialize emtpy array with size y
    tmp_x = np.zeros_like(tmp_y, dtype=np.double)

    # computing tmp_x values
    tmp_x[-1] = tmp_y[-1] / tmp_u[-1, -1]
    for i in range(n - 2, -1, -1):
        tmp_x[i] = (tmp_y[i] - np.dot(tmp_u[i, i + 1:], tmp_x[i + 1:])) / tmp_u[i, i]
    return tmp_x


def solve_lin_system_with_lu(mat, x):
    TMP_L, TMP_U = lu_decomposition(mat)
    y = forward_substitution(TMP_L, x)
    return backward_substitution(TMP_U, y)


def determinant_of_matrix(mat):
    TMP_L, TMP_U = lu_decomposition(mat)
    return np.prod(np.diag(TMP_U))


def inverse_of_matrix(mat):
    v = np.identity(len(mat))
    TMP_L, TMP_U = lu_decomposition(mat)
    TMP_L_inverse = np.transpose(np.array([forward_substitution(TMP_L, v[i]) for i in range(len(mat))]))
    TMP_U_inverse = np.transpose(np.array([backward_substitution(TMP_U, v[i]) for i in range(len(mat))]))
    return np.matmul(TMP_U_inverse, TMP_L_inverse)


# L, U = lu_decomposition(A)
# print_matrix(L, "l")
# print_matrix(U, "U")
# print_matrix(A, "A")
# print_matrix(b, "b=")
#
# result = solve_lin_system_with_lu(A, b)
# print_matrix(result, "x")
#
# det = determinant_of_matrix(A)
# print(f'det(A)={det}')
#
# inverse = inverse_of_matrix(A)
# print_matrix(inverse, "inverse")
