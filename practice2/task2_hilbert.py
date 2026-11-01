import numpy as np
from fractions import Fraction
from math import prod


def manual_det_3x3(m):
    return (
        m[0][0] * m[1][1] * m[2][2]
        + m[0][1] * m[1][2] * m[2][0]
        + m[0][2] * m[1][0] * m[2][1]
        - m[0][2] * m[1][1] * m[2][0]
        - m[0][1] * m[1][0] * m[2][2]
        - m[0][0] * m[1][2] * m[2][1]
    )


def hilbert_exact(n):
    return [[Fraction(1, i + j + 1) for j in range(n)] for i in range(n)]


def determinant_fraction(matrix):
    a = [row[:] for row in matrix]
    n = len(a)
    sign = 1
    for k in range(n - 1):
        pivot = next((i for i in range(k, n) if a[i][k] != 0), None)
        if pivot is None:
            return Fraction(0, 1)
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            sign *= -1
        value = a[k][k]
        for i in range(k + 1, n):
            factor = a[i][k] / value
            for j in range(k + 1, n):
                a[i][j] -= factor * a[k][j]
            a[i][k] = 0
    return sign * prod(a[i][i] for i in range(n))


if __name__ == "__main__":
    n = 4
    H_fraction = hilbert_exact(n)
    H64 = np.array(H_fraction, dtype=np.float64)
    H32 = np.array(H_fraction, dtype=np.float32)
    det_exact = determinant_fraction(H_fraction)
    det64 = np.linalg.det(H64)
    det32 = np.linalg.det(H32)
    inverse = np.linalg.inv(H64)
    identity_error = np.linalg.norm(H64 @ inverse - np.eye(n), ord=2)
    cond1 = np.linalg.cond(H64, p=1)
    minor = [[H_fraction[i][j] for j in range(3)] for i in range(3)]
    minor_manual = manual_det_3x3(minor)
    minor_numpy = np.linalg.det(np.array(minor, dtype=np.float64))
    print("Точный определитель:", det_exact)
    print("Определитель float64:", det64)
    print("Определитель float32:", det32)
    print("Число обусловленности по норме 1:", cond1)
    print("Норма H * H^-1 - I:", identity_error)
    print("Минор 3x3 вручную:", float(minor_manual))
    print("Минор 3x3 NumPy:", minor_numpy)
