import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.extend([current_dpath,dirname(current_dpath)])




def test_circum_center_radius():
    '''
    circum_center_radius(): boundary, near lon=0, big triangle
    '''
    from numpy import pi
    from circumcircle import circum_center_radius
    from sphere import angle
    from convert_coord.cart_ll import latlon2xyz


    # boundary
    ll1, ll2, ll3 = (0,pi/3), (0,2/3*pi), (pi/6,pi/2)
    xyz1, xyz2, xyz3 = [latlon2xyz(*ll) for ll in [ll1,ll2,ll3]]
    center, radius = circum_center_radius(xyz1, xyz2, xyz3)
    equal(center, (0,1,0))


    # near lon=0
    ll1, ll2, ll3 = (pi/5, 2*pi-pi/6), (0,pi/7), (pi/7,pi/6)
    xyz1, xyz2, xyz3 = [latlon2xyz(*ll) for ll in [ll1,ll2,ll3]]
    center, radius = circum_center_radius(xyz1, xyz2, xyz3)

    d1 = angle(center, xyz1)
    d2 = angle(center, xyz2)
    d3 = angle(center, xyz3)
    aa_equal(d1, radius, 15)
    aa_equal(d2, radius, 15)
    aa_equal(d3, radius, 15)


    # big triangle
    ll1, ll2, ll3 = (pi/2,0), (0,0), (0,pi/2)
    xyz1, xyz2, xyz3 = [latlon2xyz(*ll) for ll in [ll1,ll2,ll3]]
    center, radius = circum_center_radius(xyz1, xyz2, xyz3)
    aa_equal(center, latlon2xyz(0.61547970867038737,pi/4), 15)
