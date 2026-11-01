import time
import numpy as np
import matplotlib.pyplot as plt
from task1_methods import gauss, gauss_pivot, cramer


def gauss_count(A, b, pivoting=False):
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    n = len(b)
    operations = 0
    comparisons = 0
    for k in range(n - 1):
        if pivoting:
            pivot = k + np.argmax(np.abs(A[k:, k]))
            comparisons += n - k - 1
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        if abs(A[k, k]) < 1e-12:
            raise np.linalg.LinAlgError("Нулевой ведущий элемент")
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            operations += 1
            for j in range(k + 1, n):
                A[i, j] -= factor * A[k, j]
                operations += 2
            b[i] -= factor * b[k]
            operations += 2
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        value = b[i]
        for j in range(i + 1, n):
            value -= A[i, j] * x[j]
            operations += 2
        x[i] = value / A[i, i]
        operations += 1
    return x, operations, comparisons


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    ns = [3, 5, 10]
    cramer_times = []
    for n in ns:
        A = rng.random((n, n)) + np.eye(n)
        x_true = np.ones(n)
        b = A @ x_true
        start = time.perf_counter()
        cramer(A, b)
        cramer_times.append(time.perf_counter() - start)
        x1, count1, _ = gauss_count(A, b)
        x2, count2, comparisons = gauss_count(A, b, pivoting=True)
        print(f"n={n}: Гаусс {count1} операций; Гаусс с выбором {count2} операций; сравнений {comparisons}")
    plt.plot(ns, cramer_times, "o-")
    plt.xlabel("Размерность n")
    plt.ylabel("Время выполнения, с")
    plt.title("Время работы метода Крамера")
    plt.grid(True)
    plt.savefig("practice5_cramer_time.png", dpi=150)
    plt.show()
