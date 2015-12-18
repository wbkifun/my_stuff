#------------------------------------------------------------------------------
# filename  : cart_cs.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Development Team, KIAPS
# update    : 2014.3.12 start
#
#
# description: 
#   Convert coordinates between Cartesian and Cubed-sphere
#
# subroutines:
#   xyp2xyz()
#   xyz2xyp()
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy import pi, tan, sqrt

from ..misc.compare_float import fgt, flt, flge, feq




def xyp2xyz(x, y, panel, R=1):
    a = R/sqrt(3)
    r = sqrt(a*a + x*x + y*y)


    if panel == 1:
        X = a*R/r
        Y, Z = x/a*X, y/a*X

    elif panel == 2:
        Y = a*R/r
        X, Z = -x/a*Y, y/a*Y

    elif panel == 3:
        X = -a*R/r
        Y, Z = x/a*X, -y/a*X

    elif panel == 4:
        Y = -a*R/r
        X, Z = -x/a*Y, -y/a*Y

    elif panel == 5:
        Z = -a*R/r
        Y, X = -x/a*Z, -y/a*Z

    elif panel == 6:
        Z = a*R/r
        Y, X = x/a*Z, -y/a*Z


    return (X, Y, Z)




def xyz2xyp(X, Y, Z, R=1):
    assert feq(sqrt(X*X + Y*Y + Z*Z),R), 'The (x,y,z) (%s,%s,%s) is not on the sphere.'%(X,Y,Z)

    a = R/sqrt(3)
    at1, at2 = a*tan(-pi/4), a*tan(pi/4)

    xyp_dict = dict()


    if fgt(X,0):
        x, y = a*(Y/X), a*(Z/X)
        if flge(at1,x,at2) and flge(at1,y,at2):
            xyp_dict[1] = (x,y)

    elif flt(X,0):
        x, y = a*(Y/X), -a*(Z/X)
        if flge(at1,x,at2) and flge(at1,y,at2):
            xyp_dict[3] = (x,y)


    if fgt(Y,0):
        x, y = -a*(X/Y), a*(Z/Y)
        if flge(at1,x,at2) and flge(at1,y,at2):
            xyp_dict[2] = (x,y)

    elif flt(Y,0):
        x, y = -a*(X/Y), -a*(Z/Y)
        if flge(at1,x,at2) and flge(at1,y,at2):
            xyp_dict[4] = (x,y)


    if flt(Z,0):
        x, y = -a*(Y/Z), -a*(X/Z)
        if flge(at1,x,at2) and flge(at1,y,at2):
            xyp_dict[5] = (x,y)

    elif fgt(Z,0):
        x, y = a*(Y/Z), -a*(X/Z)
        if flge(at1,x,at2) and flge(at1,y,at2):
            xyp_dict[6] = (x,y)


    return xyp_dict
