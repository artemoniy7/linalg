import numpy as np


def gauss_inverse(A, eps=1e-14):
    A = np.asarray(A, dtype=float).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    n = A.shape[0]
    inverse = np.eye(n)
    for k in range(n):
        pivot = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[pivot, k]) < eps:
            raise np.linalg.LinAlgError("Матрица вырождена или плохо обусловлена")
        A[[k, pivot]] = A[[pivot, k]]
        inverse[[k, pivot]] = inverse[[pivot, k]]
        pivot_value = A[k, k]
        A[k] /= pivot_value
        inverse[k] /= pivot_value
        for i in range(n):
            if i != k:
                factor = A[i, k]
                A[i] -= factor * A[k]
                inverse[i] -= factor * inverse[k]
    return inverse


def svd_pseudoinverse(A, tolerance=1e-12):
    U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
    inverse_values = np.array([1 / s if s > tolerance else 0.0 for s in singular_values])
    return Vh.T @ np.diag(inverse_values) @ U.T


if __name__ == "__main__":
    A = np.array([[1, 2, 3], [2, 4, 6], [1, 0, 1]], dtype=float)
    print("Ранг вырожденной матрицы:", np.linalg.matrix_rank(A))
    try:
        np.linalg.inv(A)
    except np.linalg.LinAlgError as error:
        print("Ошибка прямого обращения:", error)
    A_near = A.copy()
    A_near[1, 1] += 1e-16
    print("Определитель почти вырожденной матрицы:", np.linalg.det(A_near))
    print("Число обусловленности:", np.linalg.cond(A_near))
    try:
        print("Обратная матрица методом Гаусса:\n", gauss_inverse(A_near))
    except np.linalg.LinAlgError as error:
        print("Метод Гаусса:", error)
    pinv = svd_pseudoinverse(A)
    print("Сингулярные числа:", np.linalg.svd(A, compute_uv=False))
    print("Проверка A * A+ * A == A:", np.allclose(A @ pinv @ A, A))
    print("Псевдообратная матрица:\n", pinv)
