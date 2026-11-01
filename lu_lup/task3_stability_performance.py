import timeit
import numpy as np
import matplotlib.pyplot as plt
from task1_algorithms import solve_via_lu, solve_via_plu, plu_decompose


def hilbert(n):
    i = np.arange(1, n + 1)[:, None]
    j = np.arange(1, n + 1)
    return 1.0 / (i + j - 1)


if __name__ == "__main__":
    for n in [5, 8, 12]:
        H = hilbert(n)
        x_true = np.ones(n)
        b = H @ x_true
        print(f"\nМатрица Гильберта {n}x{n}, cond={np.linalg.cond(H):.3e}")
        try:
            x_lu = solve_via_lu(H, b)
            print("LU: норма невязки:", np.linalg.norm(b - H @ x_lu))
        except ValueError as error:
            print("LU не выполнено:", error)
        x_lup = solve_via_plu(H, b)
        print("LUP: норма невязки:", np.linalg.norm(b - H @ x_lup))
    A_need_pivot = np.array([[0., 2., 1.], [1., 1., 0.], [2., 0., 1.]])
    b_need_pivot = A_need_pivot @ np.ones(3)
    try:
        solve_via_lu(A_need_pivot, b_need_pivot)
    except ValueError as error:
        print("LU без перестановок:", error)
    print("LUP для матрицы с нулями на диагонали:", solve_via_plu(A_need_pivot, b_need_pivot))
    rng = np.random.default_rng(42)
    sizes = [100, 500, 1000]
    time_plu, time_numpy = [], []
    for n in sizes:
        A = rng.random((n, n)) + np.eye(n)
        b = rng.random(n)
        repeats = 1 if n >= 500 else 2
        t1 = timeit.timeit(lambda: plu_decompose(A), number=repeats) / repeats
        t2 = timeit.timeit(lambda: np.linalg.solve(A, b), number=repeats) / repeats
        time_plu.append(t1)
        time_numpy.append(t2)
        print(f"n={n}: PLU {t1:.6f} с; NumPy solve {t2:.6f} с")
    plt.plot(sizes, time_plu, "o-", label="Собственная PLU")
    plt.plot(sizes, time_numpy, "o-", label="NumPy solve")
    plt.xlabel("Размер матрицы n")
    plt.ylabel("Время, с")
    plt.legend()
    plt.grid(True)
    plt.savefig("lu_lup_performance.png", dpi=150)
    plt.show()
