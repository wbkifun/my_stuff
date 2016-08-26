#------------------------------------------------------------------------------
# filename  : duplicate.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2014.3.25     start
#             2016.8.25     fix the relative import path
#
#
# description: 
#   check the duplicated points
#
# subroutines:
#   duplicate_idxs()
#   remove_duplicates()
#------------------------------------------------------------------------------

import numpy as np

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.extend([current_dpath,dirname(current_dpath)])
from misc.compare_float import feq




def duplicate_idxs(xyzs, digit=15):
    size = len(xyzs)
    dup_idxs = list()

    for i in range(size):
        for j in range(i+1,size):
            x1, y1, z1 = xyzs[i]
            x2, y2, z2 = xyzs[j]

            if feq(x1,x2,digit) and feq(y1,y2,digit) and feq(z1,z2,digit):
                dup_idxs.append(j)
    
    return dup_idxs




def remove_duplicates(xyzs, digit=15):
    dup_idxs = duplicate_idxs(xyzs, digit)

    unique_xyzs = list()
    for seq, xyz in enumerate(xyzs):
        if seq not in dup_idxs:
            unique_xyzs.append(xyz)

    return unique_xyzs
