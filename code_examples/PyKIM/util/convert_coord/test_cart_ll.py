import numpy as np
from numpy import pi
from numpy.random import rand
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.append(current_dpath)




def test_latlon2xyz():
    '''
    latlon2xyz(): center of panel, at panel border
    '''
    from cart_ll import latlon2xyz


    lat, lon = 0, 0
    aa_equal(latlon2xyz(lat, lon), (1,0,0), 15)

    lat, lon = -pi/2, 0
    aa_equal(latlon2xyz(lat, lon), (0,0,-1), 15)

    lat, lon = pi/2, 0
    aa_equal(latlon2xyz(lat, lon), (0,0,1), 15)

    lat, lon = 0, pi/2
    aa_equal(latlon2xyz(lat, lon), (0,1,0), 15)

    lat, lon = 0, pi
    aa_equal(latlon2xyz(lat, lon), (-1,0,0), 15)

    lat, lon = 0, 3*pi/2
    aa_equal(latlon2xyz(lat, lon), (0,-1,0), 15)




def test_latlon2xyz_xyz2latlon():
    '''
    latlon2xyz() -> xyz2latlon() : check consistency, repeat 1000 times
    '''
    from cart_ll import latlon2xyz, xyz2latlon


    N = 1000

    for i in range(N):
        lat = pi*rand() - pi/2
        lon = 2*pi*rand()

        X, Y, Z = latlon2xyz(lat, lon)
        lat2, lon2 = xyz2latlon(X, Y, Z)

        aa_equal((lat,lon), (lat2,lon2), 15)
