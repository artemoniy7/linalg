import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    temperature = np.array([20, 22, 25, 21, 23], dtype=float)
    energy = np.array([15, 18, 22, 16, 19], dtype=float)
    X = np.column_stack((np.ones(len(temperature)), temperature))
    y = energy
    M = X.T @ X
    det_M = np.linalg.det(M)
    w_normal = np.linalg.inv(M) @ X.T @ y
    w_pinv = np.linalg.pinv(X) @ y
    w_lstsq, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    residual_vector = y - X @ w_lstsq
    print("Матрица плана X:\n", X)
    print("Матрица моментов X.T @ X:\n", M)
    print("Определитель матрицы моментов:", det_M)
    print("Коэффициенты нормальных уравнений:", w_normal)
    print("Коэффициенты псевдообращения:", w_pinv)
    print("Коэффициенты lstsq:", w_lstsq)
    print("Ранг X:", rank)
    print("X.T @ остатки:", X.T @ residual_vector)
    order = np.argsort(temperature)
    plt.scatter(temperature, energy, label="Наблюдения")
    plt.plot(temperature[order], (X @ w_lstsq)[order], label="Линия регрессии")
    for t, e, predicted in zip(temperature, energy, X @ w_lstsq):
        plt.plot([t, t], [e, predicted])
    plt.xlabel("Температура T (°C)")
    plt.ylabel("Потребление E (кВт·ч)")
    plt.legend()
    plt.grid(True)
    plt.savefig("practice2_regression.png", dpi=150)
    plt.show()
