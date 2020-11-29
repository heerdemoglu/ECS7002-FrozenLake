import numpy as np


def derive_gmap(matrix):
    matrix = np.array(matrix)

    n1 = len(matrix)
    n2 = len(matrix[0])
    shape = [n1, n2]

    gmap = np.zeros(shape)

    for i in range(n1):
        for j in range(n2):
            if matrix[i, j] == '$':
                gmap[i, j] = 1
            else:
                gmap[i, j] = 0
    return gmap
