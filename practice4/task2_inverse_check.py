import numpy as np


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    A = rng.random((5, 5))
    inverse = np.linalg.inv(A)
    result = A @ inverse
    max_error = np.max(np.abs(result - np.eye(5)))
    print("Матрица A:\n", A)
    print("A * A^-1:\n", result)
    print("Максимальная абсолютная ошибка:", max_error)
