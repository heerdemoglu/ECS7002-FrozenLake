import numpy as np


def derive_gmap(matrix):
    matrix = np.array(matrix) # convert to numpy array

    # get dimensions
    N1 = len(matrix)
    N2 = len(matrix[0])

    # get shape and create appropriate numpy zero matrix of same shape
    shape = [N1, N2]
    gmap = np.zeros(shape)

    # store a 1 at reward position, and 0 elsewhere
    for i in range(N1):
        for j in range(N2):
            if matrix[i,j] == '$':
                gmap[i,j] = 1
            else:
                gmap[i,j] = 0
    return gmap



