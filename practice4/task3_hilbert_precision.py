import numpy as np
from sympy import Matrix, Rational


if __name__ == "__main__":
    n = 8
    H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    cond = np.linalg.cond(H, p=1)
    inverse_float = np.linalg.inv(H)
    H_exact = Matrix([[Rational(1, i + j + 1) for j in range(n)] for i in range(n)])
    inverse_exact = np.array(H_exact.inv(), dtype=float)
    difference = inverse_float - inverse_exact
    print("Число обусловленности H8 по норме 1:", cond)
    print("Максимальная поэлементная ошибка:", np.max(np.abs(difference)))
    print("Евклидова норма ошибки:", np.linalg.norm(difference, ord="fro"))
