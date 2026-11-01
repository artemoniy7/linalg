import numpy as np
from sympy import Matrix


def compute_rank_and_inverse(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("Ожидается двумерная матрица")
    rank = np.linalg.matrix_rank(A)
    if A.shape[0] != A.shape[1] or rank < A.shape[0]:
        inverse = None
    else:
        inverse = np.linalg.inv(A)
    return rank, np.linalg.det(A) if A.shape[0] == A.shape[1] else None, inverse


def verify(A, name):
    rank, det, inverse = compute_rank_and_inverse(A)
    exact = Matrix(A.tolist())
    print("---", name, "---")
    print("Ранг:", rank)
    print("Определитель:", det)
    if inverse is None:
        print("Обратная матрица: отсутствует")
    else:
        print("Обратная матрица:\n", inverse)
        exact_inverse = np.array(exact.inv(), dtype=float)
        print("Норма ошибки обратной матрицы:", np.linalg.norm(inverse - exact_inverse, ord="fro"))


if __name__ == "__main__":
    E = np.eye(3)
    P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    A = np.array([[2, -1, 0], [1, 2, 1], [0, 3, 4]], dtype=float)
    verify(E, "Единичная матрица")
    verify(A, "Невырожденная матрица")
    verify(P, "Матрица перестановки")
