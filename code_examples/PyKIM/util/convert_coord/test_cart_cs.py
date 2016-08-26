import numpy as np
from numpy import pi, sin, cos, tan, sqrt
from numpy.random import rand, randint
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.append(current_dpath)




def test_xyp2xyz():
    '''
    xyp2xyz(): center of panel, at panel border
    '''
    from cart_cs import xyp2xyz

    R = 1
    a = R/sqrt(3)


    #------------------------------------------------
    # center of panel
    #------------------------------------------------
    x, y, panel = 0, 0, 1
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (1,0,0))

    x, y, panel = 0, 0, 2
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (0,1,0))

    x, y, panel = 0, 0, 3
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (-1,0,0))

    x, y, panel = 0, 0, 4
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (0,-1,0))

    x, y, panel = 0, 0, 5
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (0,0,-1))

    x, y, panel = 0, 0, 6
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (0,0,1))


    #------------------------------------------------
    # at the panel border
    #------------------------------------------------
    alpha = pi/4

    x, y, panel = a*tan(alpha), 0, 1
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (R*cos(alpha), R*sin(alpha), 0))

    x, y, panel = a*tan(alpha), 0, 2
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (-R*sin(alpha), R*cos(alpha), 0))

    x, y, panel = 0, -a*tan(alpha), 2
    (X, Y, Z) = xyp2xyz(x, y, panel)
    aa_equal((X,Y,Z), (0, R*sin(alpha), -R*cos(alpha)), 15)


    x, y, panel = a*tan(alpha), 0, 3
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (-R*cos(alpha), -R*sin(alpha), 0))

    x, y, panel = a*tan(alpha), 0, 4
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (R*sin(alpha), -R*cos(alpha), 0))

    x, y, panel = a*tan(alpha), 0, 5
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (0, R*sin(alpha), -R*cos(alpha)))

    x, y, panel = a*tan(alpha), 0, 6
    (X, Y, Z) = xyp2xyz(x, y, panel)
    a_equal((X,Y,Z), (0, R*sin(alpha), R*cos(alpha)))




def test_xyz2xyp():
    '''
    xyz2xyp(): center of panel, at panel border
    '''
    from cart_cs import xyz2xyp

    R = 1
    a = R/sqrt(3)


    #------------------------------------------------
    # center of panel
    #------------------------------------------------
    xyp_dict = xyz2xyp(1, 0, 0)
    a_equal(xyp_dict, {1:(0.0,0)})

    xyp_dict = xyz2xyp(0, 1, 0)
    a_equal(xyp_dict, {2:(0,0)})

    xyp_dict = xyz2xyp(-1, 0, 0)
    a_equal(xyp_dict, {3:(0,0)})

    xyp_dict = xyz2xyp(0, -1, 0)
    a_equal(xyp_dict, {4:(0,0)})

    xyp_dict = xyz2xyp(0, 0, -1)
    a_equal(xyp_dict, {5:(0,0)})

    xyp_dict = xyz2xyp(0, 0, 1)
    a_equal(xyp_dict, {6:(0,0)})


    #------------------------------------------------
    # at the panel border
    #------------------------------------------------
    alpha = pi/4
    at = a*tan(alpha)

    xyp_dict = xyz2xyp(R*cos(alpha), R*sin(alpha), 0)
    a_equal(list(xyp_dict.keys()), [1,2])
    aa_equal(list(xyp_dict.values()), [(at,0), (-at,0)], 15)

    xyp_dict = xyz2xyp(-R*sin(alpha), R*cos(alpha), 0)
    a_equal(list(xyp_dict.keys()), [2,3])
    aa_equal(list(xyp_dict.values()), [(at,0), (-at,0)], 15)

    xyp_dict = xyz2xyp(-R*cos(alpha), -R*sin(alpha), 0)
    a_equal(list(xyp_dict.keys()), [3,4])
    aa_equal(list(xyp_dict.values()), [(at,0), (-at,0)], 15)

    xyp_dict = xyz2xyp(R*sin(alpha), -R*cos(alpha), 0)
    a_equal(list(xyp_dict.keys()), [1,4])
    aa_equal(list(xyp_dict.values()), [(-at,0), (at,0)], 15)

    xyp_dict = xyz2xyp(0, R*sin(alpha), -R*cos(alpha))
    a_equal(list(xyp_dict.keys()), [2,5])
    aa_equal(list(xyp_dict.values()), [(0,-at), (at,0)], 15)

    xyp_dict = xyz2xyp(0, R*sin(alpha), R*cos(alpha))
    a_equal(list(xyp_dict.keys()), [2,6])
    aa_equal(list(xyp_dict.values()), [(0,at), (at,0)], 15)




def test_xyp2xyz_xyz2xyp():
    '''
    xyp2xyz() -> xyz2xyp() : check consistency, repeat 1000 times
    '''
    from cart_cs import xyp2xyz, xyz2xyp


    N = 1000
    R = 1
    a = R/sqrt(3)

    for i in range(N):
        panel = randint(1,7)
        alpha, beta = (pi/2)*rand(2) - pi/4
        x, y = a*tan(alpha), a*tan(beta)

        (X, Y, Z) = xyp2xyz(x, y, panel)
        xyp_dict = xyz2xyp(X,Y,Z)

        aa_equal((x,y), xyp_dict[panel], 15)
