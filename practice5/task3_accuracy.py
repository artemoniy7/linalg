import numpy as np
from task1_methods import cramer, gauss, gauss_pivot, generate_hilbert


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for n in [3, 5, 10]:
        A = rng.random((n, n)) + np.eye(n)
        x_true = np.ones(n)
        b = A @ x_true
        print(f"\nСлучайная матрица {n}x{n}")
        print("Число обусловленности:", np.linalg.cond(A, p=2))
        for name, solver in [("Крамер", cramer), ("Гаусс", gauss), ("Гаусс с выбором", gauss_pivot)]:
            try:
                x = solver(A, b)
                residual = b - A @ x
                print(name, "решение:", x)
                print(name, "норма невязки:", np.linalg.norm(residual))
                print(name, "максимальная невязка:", np.max(np.abs(residual)))
                print(name, "ошибка решения:", np.linalg.norm(x_true - x))
            except np.linalg.LinAlgError as error:
                print(name, "ошибка:", error)
    for n in [5, 8, 12]:
        H = generate_hilbert(n)
        x_true = np.ones(n)
        b = H @ x_true
        print(f"\nМатрица Гильберта {n}x{n}, cond={np.linalg.cond(H):.3e}")
        for name, solver in [("Гаусс", gauss), ("Гаусс с выбором", gauss_pivot)]:
            try:
                x = solver(H, b)
                print(name, "норма невязки:", np.linalg.norm(b - H @ x))
            except np.linalg.LinAlgError as error:
                print(name, "ошибка:", error)
