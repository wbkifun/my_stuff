import numpy as np
from numpy import pi
from numpy.random import rand, randint
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.append(current_dpath)




def test_xyz_rotate():
    '''
    xyz_rotate()
    '''
    from cart_rotate import xyz_rotate


    x, y, z = xyz_rotate((1,0,0), rlat=0, rlon=pi/2)
    aa_equal((x,y,z), (0,-1,0), 15)

    x, y, z = xyz_rotate((1,0,0), rlat=0, rlon=pi)
    aa_equal((x,y,z), (-1,0,0), 15)

    x, y, z = xyz_rotate((1,0,0), rlat=0, rlon=3*pi/2)
    aa_equal((x,y,z), (0,1,0), 15)

    x, y, z = xyz_rotate((1,0,0), rlat=pi/2, rlon=0)
    aa_equal((x,y,z), (0,0,-1), 15)

    x, y, z = xyz_rotate((1,0,0), rlat=-pi/2, rlon=0)
    aa_equal((x,y,z), (0,0,1), 15)




def test_xyp_rotate_reverse():
    '''
    xyp_rotate() -> xyz_reverse_rotate() : check consistency, repeat 1000 times
    '''
    from cart_rotate import xyz_rotate, xyz_rotate_reverse
    from cart_ll import latlon2xyz


    N = 1000

    for i in range(N):
        lat = pi*rand() - pi/2
        lon = 2*pi*rand()
        x, y, z = latlon2xyz(lat, lon)

        rlat = pi*rand() - pi/2
        rlon = 2*pi*rand()
        xr, yr, zr = xyz_rotate((x, y, z), rlat, rlon)
        x2, y2, z2 = xyz_rotate_reverse((xr, yr, zr), rlat, rlon)

        aa_equal((x,y,z), (x2,y2,z2), 15)
