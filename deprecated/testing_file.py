
#THIS FILE IS JUST FOR ROUGH TESTING, NOT NEEDED FOR PROJECT
import numpy as np
from Auxilary_Functions import derive_gmap

lake = [['&', '.', '.', '.'],
        ['.', '#', '0', '#'],
        ['.', '.', '.', '#'],
        ['#', '.', '.', '$']]

gmap = derive_gmap(lake)
print(gmap)
