#------------------------------------------------------------------------------
# filename  : cart_ll.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Development Team, KIAPS
# update    : 2014.3.12 start
#
#
# description: 
#   Convert coordinates between Cartesian and Latlon
#
# subroutines:
#   latlon2xyz()
#   xyz2latlon()
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy import pi, sin, cos, arctan, sqrt

from ..misc.compare_float import feq, fgt




def latlon2xyz(lat, lon, R=1):
    x = R*cos(lat)*cos(lon)
    y = R*cos(lat)*sin(lon)
    z = R*sin(lat)

    return x, y, z




def xyz2latlon(x, y, z, R=1):
    '''
    if feq(z,R):
        lat, lon = pi/2, 0
    elif feq(z,-R):
        lat, lon = -pi/2, 0
    '''

    if feq(sqrt(x*x + y*y),0):
        if z > 0:
            lat, lon = pi/2, 0
        elif z < 0:
            lat, lon = -pi/2, 0

    else:
        lat = arctan(z/sqrt(x*x + y*y))

        if feq(x,0):
            if fgt(y,0): lon = pi/2
            else: lon = 3*pi/2

        elif fgt(x,0):
            if fgt(y,0): lon = arctan(y/x)
            else: lon = arctan(y/x) + 2*pi

        else:
            lon = arctan(y/x) + pi

    return lat, lon
