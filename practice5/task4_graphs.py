import time
import numpy as np
import matplotlib.pyplot as plt
from task1_methods import gauss, gauss_pivot, generate_hilbert
from task2_operations import gauss_count


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dimensions = [10, 20, 30, 40, 50]
    time_plain, time_pivot, flops_plain, flops_pivot = [], [], [], []
    conditions, residuals_plain, residuals_pivot = [], [], []
    for n in dimensions:
        A = rng.random((n, n)) + np.eye(n) * n
        x_true = np.ones(n)
        b = A @ x_true
        start = time.perf_counter()
        gauss(A, b)
        time_plain.append(time.perf_counter() - start)
        start = time.perf_counter()
        gauss_pivot(A, b)
        time_pivot.append(time.perf_counter() - start)
        _, f1, _ = gauss_count(A, b)
        _, f2, _ = gauss_count(A, b, pivoting=True)
        flops_plain.append(f1)
        flops_pivot.append(f2)
        x1 = gauss(A, b)
        x2 = gauss_pivot(A, b)
        conditions.append(np.linalg.cond(A, p=2))
        residuals_plain.append(np.linalg.norm(b - A @ x1))
        residuals_pivot.append(np.linalg.norm(b - A @ x2))
    plt.plot(dimensions, time_plain, "o-", label="Гаусс")
    plt.plot(dimensions, time_pivot, "o-", label="Гаусс с выбором")
    plt.xlabel("Размерность n")
    plt.ylabel("Время, с")
    plt.legend()
    plt.grid(True)
    plt.savefig("practice5_time.png", dpi=150)
    plt.show()
    plt.plot(dimensions, flops_plain, "o-", label="Гаусс")
    plt.plot(dimensions, flops_pivot, "o-", label="Гаусс с выбором")
    plt.xlabel("Размерность n")
    plt.ylabel("Количество операций")
    plt.legend()
    plt.grid(True)
    plt.savefig("practice5_flops.png", dpi=150)
    plt.show()
    plt.scatter(conditions, residuals_plain, label="Гаусс")
    plt.scatter(conditions, residuals_pivot, label="Гаусс с выбором")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Число обусловленности")
    plt.ylabel("Норма невязки")
    plt.legend()
    plt.grid(True)
    plt.savefig("practice5_residuals.png", dpi=150)
    plt.show()
    hilbert_ns = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    hilbert_conditions = [np.linalg.cond(generate_hilbert(n), p=2) for n in hilbert_ns]
    plt.plot(hilbert_ns, hilbert_conditions, "o-")
    plt.yscale("log")
    plt.xlabel("Размерность матрицы Гильберта n")
    plt.ylabel("Число обусловленности")
    plt.grid(True)
    plt.savefig("practice5_hilbert_condition.png", dpi=150)
    plt.show()
