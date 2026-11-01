import numpy as np


def safe_inverse(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("ERR_INVALID_DIMENSIONS: Ожидается двумерный массив")
    if np.isnan(A).any():
        raise ValueError("ERR_NAN_INPUT: Входные данные содержат NaN")
    if np.isinf(A).any():
        raise ValueError("ERR_INF_INPUT: Входные данные содержат Inf")
    if A.shape[0] != A.shape[1]:
        return {"rank": int(np.linalg.matrix_rank(A)), "inverse": None, "status": "ERR_NONSQUARE"}
    rank = int(np.linalg.matrix_rank(A))
    if rank < A.shape[0]:
        return {"rank": rank, "inverse": None, "status": "ERR_SINGULAR"}
    condition = np.linalg.cond(A)
    if condition > 1e14:
        return {"rank": rank, "inverse": None, "status": "WARN_ILL_CONDITIONED"}
    return {"rank": rank, "inverse": np.linalg.inv(A), "status": "OK"}


if __name__ == "__main__":
    tests = {
        "Нулевая матрица": np.zeros((3, 3)),
        "Вектор-столбец": np.array([[1], [2], [3]]),
        "Строчная матрица": np.array([[1, 2, 3]]),
        "NaN": np.array([[1, np.nan], [0, 1]]),
        "Inf": np.array([[1, np.inf], [0, 1]]),
        "Обычная матрица": np.array([[2, 1], [1, 2]]),
    }
    for name, matrix in tests.items():
        try:
            result = safe_inverse(matrix)
        except ValueError as error:
            result = str(error)
        print(name, "->", result)
