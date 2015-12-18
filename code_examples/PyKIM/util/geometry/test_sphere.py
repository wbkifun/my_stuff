#------------------------------------------------------------------------------
# filename  : test_sphere.py
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




def test_angle():
    '''
    angle(): yz plane circle, oblique circle
    '''

    from sphere import angle
    from ..convert_coord.cart_ll import latlon2xyz


    ret = angle(latlon2xyz(pi/4,0), latlon2xyz(0,0))
    equal(ret, pi/4)

    ret = angle(latlon2xyz(pi/2,0), latlon2xyz(0,0))
    equal(ret, pi/2)

    ret = angle(latlon2xyz(0,0), latlon2xyz(0,0))
    equal(ret, 0)

    ret = angle(latlon2xyz(pi/4,0), latlon2xyz(-pi/4,0))
    aa_equal(ret, pi/2, 15)

    ret = angle(latlon2xyz(pi/2,0), latlon2xyz(-pi/4,0))
    aa_equal(ret, 3*pi/4, 15)

    ret = angle(latlon2xyz(pi/2,0), latlon2xyz(-pi/2,0))
    aa_equal(ret, pi, 15)




def test_area_polygon():
    '''
    area_polygon(): 1/8, 1/16, 1/24, 1/48
    '''

    from sphere import area_polygon
    from ..convert_coord.cart_ll import latlon2xyz
    from ..convert_coord.cs_ll import abp2latlon


    # 1/8 sphere area
    latlons = [(0,0), (0,pi/2), (pi/2,0)]
    xyzs = [latlon2xyz(*latlon) for latlon in latlons]
    ret = area_polygon(xyzs)
    aa_equal(ret, 4*pi/8, 15)


    # 1/16 sphere area
    latlons = [(0,0), (0,pi/4), (pi/2,0)]
    xyzs = [latlon2xyz(*latlon) for latlon in latlons]
    ret = area_polygon(xyzs)
    aa_equal(ret, 4*pi/16, 15)


    # 1/24 sphere area
    latlon1 = (0,0)
    latlon2 =  abp2latlon(-pi/4,0,1)
    latlon3 =  abp2latlon(-pi/4,-pi/4,1)
    latlon4 =  abp2latlon(0,-pi/4,1) 
    latlons = [latlon1, latlon2, latlon3, latlon4]
    xyzs = [latlon2xyz(*latlon) for latlon in latlons]
    ret = area_polygon(xyzs)
    aa_equal(ret, 4*pi/24, 15)


    # 1/48 sphere area
    latlons = [latlon1, latlon3, latlon4]
    xyzs = [latlon2xyz(*latlon) for latlon in latlons]
    ret = area_polygon(xyzs)
    aa_equal(ret, 4*pi/48, 15)




def test_intersect_two_greatcircles():
    '''
    intersect_two_greatcircles(): axis circles, oblique circles, identical
    '''

    from sphere import plane_origin, intersect_two_greatcircles
    from ..convert_coord.cart_ll import latlon2xyz


    #---------------------------------------
    # axis circles
    #---------------------------------------
    xyz1 = (1,0,0)
    xyz2 = (0,1,0)
    xyz3 = (0,0,1)

    # x axis
    plane1 = plane_origin(xyz1, xyz2)
    plane2 = plane_origin(xyz1, xyz3)
    ret = intersect_two_greatcircles(plane1, plane2)
    equal(ret, [(1,0,0), (-1,0,0)])

    # y axis
    plane1 = plane_origin(xyz1, xyz2)
    plane2 = plane_origin(xyz2, xyz3)
    ret = intersect_two_greatcircles(plane1, plane2)
    equal(ret, [(0,1,0), (0,-1,0)])

    # z axis
    plane1 = plane_origin(xyz1, xyz3)
    plane2 = plane_origin(xyz2, xyz3)
    ret = intersect_two_greatcircles(plane1, plane2)
    equal(ret, [(0,0,1), (0,0,-1)])


    #---------------------------------------
    # oblique circles
    #---------------------------------------
    xyz1 = (0, 0, 1)
    xyz2 = latlon2xyz(pi/4, pi/4)
    xyz3 = (1,0,0)

    plane1 = plane_origin(xyz1, xyz2)
    plane2 = plane_origin(xyz2, xyz3)

    ret = intersect_two_greatcircles(plane1, plane2)
    aa_equal(ret, [xyz2, latlon2xyz(-pi/4, 5*pi/4)], 15)


    #---------------------------------------
    # identical
    #---------------------------------------
    xyz1 = (0, 0, 1)
    xyz2 = latlon2xyz(pi/4, pi/4)
    plane = plane_origin(xyz1, xyz2)
    ret = intersect_two_greatcircles(plane, plane)
    equal(ret, [None, None])
