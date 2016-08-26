#------------------------------------------------------------------------------
# filename  : cart_cs.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Development Team, KIAPS
# update    : 2014.3.12 start
#
#
# description: 
#   Rotate Cartesian coordinates
#
# subroutines:
#   xyz_rotate()
#   xyz_rotate_reverse()
#------------------------------------------------------------------------------

import numpy as np
from numpy import sin, cos




def xyz_rotate(xyz, rlat, rlon):
    x, y, z = xyz
    xr = cos(rlat)*cos(rlon)*x + cos(rlat)*sin(rlon)*y + sin(rlat)*z
    yr = -sin(rlon)*x + cos(rlon)*y
    zr = -sin(rlat)*cos(rlon)*x - sin(rlat)*sin(rlon)*y + cos(rlat)*z

    return (xr, yr, zr)




def xyz_rotate_reverse(xyz, rlat, rlon):
    x, y, z = xyz
    xr = cos(rlat)*cos(rlon)*x - sin(rlon)*y - sin(rlat)*cos(rlon)*z
    yr = cos(rlat)*sin(rlon)*x + cos(rlon)*y - sin(rlat)*sin(rlon)*z
    zr = sin(rlat)*x + cos(rlat)*z

    return (xr, yr, zr)
