import numpy as np


def lu_decompose(A, eps=1e-18):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    for k in range(n):
        if abs(U[k, k]) < eps:
            raise ValueError("матрица не подходит для разложения без перестановок")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return L, U


def plu_decompose(A, eps=1e-18):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    piv = np.arange(n)
    for k in range(n):
        pivot = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[pivot, k]) < eps:
            raise np.linalg.LinAlgError("Матрица вырождена")
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            piv[[k, pivot]] = piv[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return piv, L, U


def forward_substitution(L, b):
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        if abs(L[i, i]) < 1e-18:
            raise np.linalg.LinAlgError("Нулевой диагональный элемент L")
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def backward_substitution(U, y):
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-18:
            raise np.linalg.LinAlgError("Нулевой диагональный элемент U")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve_via_lu(A, b):
    L, U = lu_decompose(A)
    return backward_substitution(U, forward_substitution(L, b))


def solve_via_plu(A, b):
    piv, L, U = plu_decompose(A)
    y = forward_substitution(L, np.asarray(b, dtype=float)[piv])
    return backward_substitution(U, y)


if __name__ == "__main__":
    A = np.array([[2., 1., 1.], [4., -6., 0.], [-2., 7., 2.]])
    b = np.array([5., -2., 9.])
    print("Решение LU:", solve_via_lu(A, b))
    print("Решение LUP:", solve_via_plu(A, b))
    print("Проверка LUP:", A @ solve_via_plu(A, b))
