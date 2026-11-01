import numpy as np
from scipy.linalg import lu
from task1_algorithms import plu_decompose, solve_via_plu


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    A = rng.random((4, 4)) + np.eye(4)
    b = rng.random(4)
    piv, L, U = plu_decompose(A)
    P = np.eye(4)[piv]
    print("Проверка P @ A == L @ U:", np.allclose(P @ A, L @ U))
    x_true = np.ones(4)
    b_true = A @ x_true
    x_plu = solve_via_plu(A, b_true)
    x_numpy = np.linalg.solve(A, b_true)
    x_inverse = np.linalg.inv(A) @ b_true
    for name, x in [("PLU", x_plu), ("NumPy solve", x_numpy), ("Обратная матрица", x_inverse)]:
        print(name, "решение:", x)
        print(name, "ошибка относительно истинного решения:", np.linalg.norm(x - x_true))
        print(name, "норма невязки:", np.linalg.norm(b_true - A @ x))
    P_scipy, L_scipy, U_scipy = lu(A)
    print("SciPy: P @ A == L @ U:", np.allclose(P_scipy @ A, L_scipy @ U_scipy))
    print("Норма разности L:", np.linalg.norm(L - L_scipy))
    print("Норма разности U:", np.linalg.norm(U - U_scipy))
