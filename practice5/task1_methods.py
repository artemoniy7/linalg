import numpy as np


def validate_system(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or b.shape != (A.shape[0],):
        raise ValueError("Некорректные размеры A и b")
    return A.copy(), b.copy()


def cramer(A, b):
    A, b = validate_system(A, b)
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-12:
        raise np.linalg.LinAlgError("Матрица вырождена")
    x = np.zeros(len(b))
    for i in range(len(b)):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A
    return x


def gauss(A, b, eps=1e-12):
    A, b = validate_system(A, b)
    n = len(b)
    for k in range(n - 1):
        if abs(A[k, k]) < eps:
            raise np.linalg.LinAlgError("Нулевой ведущий элемент")
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k + 1:] -= factor * A[k, k + 1:]
            b[i] -= factor * b[k]
            A[i, k] = 0.0
    if abs(A[-1, -1]) < eps:
        raise np.linalg.LinAlgError("Нулевой ведущий элемент")
    return backward_substitution(A, b, eps)


def gauss_pivot(A, b, eps=1e-12):
    A, b = validate_system(A, b)
    n = len(b)
    for k in range(n - 1):
        pivot = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[pivot, k]) < eps:
            raise np.linalg.LinAlgError("Матрица вырождена или близка к ней")
        A[[k, pivot]] = A[[pivot, k]]
        b[[k, pivot]] = b[[pivot, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k + 1:] -= factor * A[k, k + 1:]
            b[i] -= factor * b[k]
            A[i, k] = 0.0
    if abs(A[-1, -1]) < eps:
        raise np.linalg.LinAlgError("Матрица вырождена")
    return backward_substitution(A, b, eps)


def backward_substitution(U, y, eps=1e-12):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < eps:
            raise np.linalg.LinAlgError("Нулевой диагональный элемент")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def count_operations_gauss(n):
    divisions = n * (n - 1) // 2 + n
    multiplications = sum((n - k - 1) ** 2 for k in range(n - 1)) + n * (n - 1) // 2
    additions = multiplications
    return {"умножения": multiplications, "деления": divisions, "сложения_вычитания": additions,
            "всего": multiplications + divisions + additions}


def generate_hilbert(n):
    i = np.arange(1, n + 1)[:, None]
    j = np.arange(1, n + 1)
    return 1.0 / (i + j - 1)


if __name__ == "__main__":
    A = np.array([[4.0, 2.0, 1.0], [1.0, 5.0, 2.0], [2.0, 1.0, 4.0]])
    x_true = np.array([2.0, -1.0, 3.0])
    b = A @ x_true
    print("Заранее известное решение:", x_true)
    print("Крамер:", cramer(A, b))
    print("Гаусс:", gauss(A, b))
    print("Гаусс с выбором главного элемента:", gauss_pivot(A, b))
    print("Операции Гаусса для n=3:", count_operations_gauss(3))
