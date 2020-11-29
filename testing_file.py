
#THIS FILE IS JUST FOR ROUGH TESTING, NOT NEEDED FOR PROJECT
import numpy as np

lake = [['&', '.', '.', '.'],
        ['.', '#', '0', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

def derive_gmap(matrix):
    matrix = np.array(matrix)

    N1 = len(matrix)
    N2 = len(matrix[0])
    shape = [N1, N2]

    gmap = np.zeros(shape)

    for i in range(N1):
        for j in range(N2):
            if matrix[i,j] == '$':
                gmap[i,j] = 1
            else:
                gmap[i,j] = 0
    return gmap

gmap = derive_gmap(lake)
print(gmap)
