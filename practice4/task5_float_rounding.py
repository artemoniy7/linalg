import numpy as np
from fractions import Fraction


if __name__ == "__main__":
    A = np.array([[0.1, 0.2], [0.3, 0.7]], dtype=np.float64)
    inverse = np.linalg.inv(A)
    product = A @ inverse
    error = product - np.eye(2)
    print("Исходная матрица:\n", A)
    print("Произведение A * A^-1:\n", product)
    print("Абсолютная ошибка:\n", error)
    print("Максимальная ошибка:", np.max(np.abs(error)))
    print("Точное представление 0.1 как дроби:", Fraction(0.1))
    print("Десятичное значение 0.1 как точная дробь:", Fraction("0.1"))
    exact = [[Fraction("0.1"), Fraction("0.2")], [Fraction("0.3"), Fraction("0.7")]]
    determinant = exact[0][0] * exact[1][1] - exact[0][1] * exact[1][0]
    inverse_exact = [
        [exact[1][1] / determinant, -exact[0][1] / determinant],
        [-exact[1][0] / determinant, exact[0][0] / determinant],
    ]
    print("Точная обратная матрица (Fraction):")
    for row in inverse_exact:
        print(row)
