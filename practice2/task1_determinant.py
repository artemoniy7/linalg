import numpy as np


def manual_det_3x3(m):
    if len(m) != 3 or any(len(row) != 3 for row in m):
        raise ValueError("Матрица должна иметь размер 3x3")
    return (
        m[0][0] * m[1][1] * m[2][2]
        + m[0][1] * m[1][2] * m[2][0]
        + m[0][2] * m[1][0] * m[2][1]
        - m[0][2] * m[1][1] * m[2][0]
        - m[0][1] * m[1][0] * m[2][2]
        - m[0][0] * m[1][2] * m[2][1]
    )


if __name__ == "__main__":
    A = [[1, 2, 1], [2, 5, 3], [3, 7, 4]]
    det_manual = manual_det_3x3(A)
    det_numpy = np.linalg.det(np.array(A, dtype=np.float64))
    A_swapped = [A[1], A[0], A[2]]
    det_swapped = manual_det_3x3(A_swapped)
    print("Определитель вручную:", det_manual)
    print("Определитель NumPy:", det_numpy)
    print("Совпадение с учетом погрешности:", np.isclose(det_manual, det_numpy, atol=1e-9))
    print("Определитель после перестановки двух строк:", det_swapped)
    print("Знак изменился:", np.isclose(det_swapped, -det_manual))
