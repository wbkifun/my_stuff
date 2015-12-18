#------------------------------------------------------------------------------
# filename  : test_area_sphere.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.17    revision from pygecore test
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy import pi, sqrt, sin
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_angle_sphere():
    '''
    angle_sphere(): yz plane circle, oblique circle
    '''

    from area_sphere import angle_sphere
    from pkg.convert_coord.cart_ll import latlon2xyz


    ret = angle_sphere(latlon2xyz(pi/4,0), latlon2xyz(0,0))
    equal(ret, pi/4)

    ret = angle_sphere(latlon2xyz(pi/2,0), latlon2xyz(0,0))
    equal(ret, pi/2)

    ret = angle_sphere(latlon2xyz(0,0), latlon2xyz(0,0))
    equal(ret, 0)

    ret = angle_sphere(latlon2xyz(pi/4,0), latlon2xyz(-pi/4,0))
    aa_equal(ret, pi/2, 15)

    ret = angle_sphere(latlon2xyz(pi/2,0), latlon2xyz(-pi/4,0))
    aa_equal(ret, 3*pi/4, 15)

    ret = angle_sphere(latlon2xyz(pi/2,0), latlon2xyz(-pi/2,0))
    aa_equal(ret, pi, 15)




def test_area_polygon_sphere():
    '''
    area_polygon_sphere(): 1/8, 1/16, 1/24, 1/48
    '''

    from area_sphere import area_polygon_sphere
    from pkg.convert_coord.cart_ll import latlon2xyz
    from pkg.convert_coord.cs_ll import abp2latlon


    # 1/8 sphere area
    latlons = [(0,0), (0,pi/2), (pi/2,0)]
    xyzs = [latlon2xyz(*latlon) for latlon in latlons]
    ret = area_polygon_sphere(xyzs)
    aa_equal(ret, 4*pi/8, 15)


    # 1/16 sphere area
    latlons = [(0,0), (0,pi/4), (pi/2,0)]
    xyzs = [latlon2xyz(*latlon) for latlon in latlons]
    ret = area_polygon_sphere(xyzs)
    aa_equal(ret, 4*pi/16, 15)


    # 1/24 sphere area
    latlon1 = (0,0)
    latlon2 =  abp2latlon(-pi/4,0,1)
    latlon3 =  abp2latlon(-pi/4,-pi/4,1)
    latlon4 =  abp2latlon(0,-pi/4,1) 
    latlons = [latlon1, latlon2, latlon3, latlon4]
    xyzs = [latlon2xyz(*latlon) for latlon in latlons]
    ret = area_polygon_sphere(xyzs)
    aa_equal(ret, 4*pi/24, 15)


    # 1/48 sphere area
    latlons = [latlon1, latlon3, latlon4]
    xyzs = [latlon2xyz(*latlon) for latlon in latlons]
    ret = area_polygon_sphere(xyzs)
    aa_equal(ret, 4*pi/48, 15)
