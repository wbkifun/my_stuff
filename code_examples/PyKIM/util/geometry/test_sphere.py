#------------------------------------------------------------------------------
# filename  : test_sphere.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.17    revision from pygecore test
#             2016.8.25     fix the relative import path
#------------------------------------------------------------------------------

import numpy as np
from numpy import pi, sqrt, sin
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.extend([current_dpath,dirname(current_dpath)])




def test_angle():
    '''
    angle(): yz plane circle, oblique circle
    '''

    from sphere import angle
    from convert_coord.cart_ll import latlon2xyz


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
    from convert_coord.cart_ll import latlon2xyz
    from convert_coord.cs_ll import abp2latlon


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




def test_normal_vector():
    '''
    normal_vector(): yz plane circle, oblique circle
    '''

    from sphere import normal_vector
    from convert_coord.cart_ll import latlon2xyz


    #---------------------------------------
    # yz plane circle, +x direction
    #---------------------------------------
    vec1 = (0, 1, 0)
    vec2 = (0, 1/sqrt(2), 1/sqrt(2))
    nvec = normal_vector(vec1, vec2)
    equal(nvec, (sin(pi/4),0,0))

    unit_nvec = normal_vector(vec1, vec2, normalize=True)
    equal(unit_nvec, (1,0,0))


    #---------------------------------------
    # yz plane circle, -x direction
    #---------------------------------------
    vec1 = (0, -1, 0)
    vec2 = (0, 1/sqrt(2), 1/sqrt(2))
    nvec = normal_vector(vec1, vec2)
    equal(nvec, (-sin(pi/4),0,0))

    unit_nvec = normal_vector(vec1, vec2, normalize=True)
    equal(unit_nvec, (-1,0,0))


    #---------------------------------------
    # oblique circle
    #---------------------------------------
    vec1 = (0, 0, 1)
    vec2 = latlon2xyz(pi/4,pi/4)
    nvec = normal_vector(vec1, vec2)
    aa_equal(nvec, latlon2xyz(0,3*pi/4,R=sin(pi/4)), 15)

    unit_nvec = normal_vector(vec1, vec2, normalize=True)
    aa_equal(unit_nvec, (-1/sqrt(2),1/sqrt(2),0), 15)




def test_sort_ccw_idxs():
    '''
    sort_ccw_idxs(): normal case, straight line, duplicated
    '''

    from sphere import sort_ccw_idxs
    from duplicate import remove_duplicates
    from convert_coord.cart_ll import latlon2xyz


    # normal
    lls = [(0.79,0.79), (0.78,0.77), (0.78,0.79), (0.79,0.77), (0.80,0.78)]
    xyzs = [latlon2xyz(*ll) for ll in lls]
    ret = sort_ccw_idxs(xyzs)
    a_equal(ret, [0,4,3,1,2])


    # straight line
    lls = [(0.79,0.79), (0.78,0.77), (0.78,0.79), \
           (0.79,0.77), (0.80,0.78), (0.78,0.78)]
    xyzs = [latlon2xyz(*ll) for ll in lls]
    ret = sort_ccw_idxs(xyzs)
    a_equal(ret, [0,4,3,1,5,2])


    #-----------------------------------------------------
    # duplicated
    #-----------------------------------------------------
    lls = [(-0.34784230590688509, 6.1959188445798699), 
           (-0.3478423059068852,  0.08726646259971646),
           (-0.52194946399942688, 0.08726646259971646),
           (-0.52194946399942688, 6.1959188445798699),
           (-0.52194946399942688, 6.1959188445798699)]
    xyzs = [latlon2xyz(*ll) for ll in lls]
    unique_xyzs = remove_duplicates(xyzs)
    ret = sort_ccw_idxs(unique_xyzs)
    a_equal(ret, [0,3,2,1])

    #-----------------------------------------------------
    lls = [(-1.3956102462281967, 0.43633231299858227),
           (-1.3956102462281967, 0.26179938779914985),
           (-1.5707963267948966, 0),
           (-1.5707963267948966, 0)]
    xyzs = [latlon2xyz(*ll) for ll in lls]
    unique_xyzs = remove_duplicates(xyzs)
    ret = sort_ccw_idxs(unique_xyzs)
    a_equal(ret, [0,1,2])




def test_arc12_pt3():
    '''
    arc12_pt3(): between, out, pt1, pt2
    '''

    from sphere import arc12_pt3
    from convert_coord.cart_ll import latlon2xyz


    #---------------------------------------
    # oblique circle
    #---------------------------------------
    vec1 = (0, 0, 1)
    vec2 = latlon2xyz(pi/4, pi/4)


    # between
    vec3 = latlon2xyz(2*pi/6, pi/4)
    ret = arc12_pt3(vec1, vec2, vec3)
    equal(ret, 'between')


    # out
    vec3 = latlon2xyz(-pi/16, pi/4)
    ret = arc12_pt3(vec1, vec2, vec3)
    equal(ret, 'out')

    vec3 = latlon2xyz(-pi/2, pi/4)
    ret = arc12_pt3(vec1, vec2, vec3)
    equal(ret, 'out')

    vec3 = latlon2xyz(2*pi/6, 5*pi/4)
    ret = arc12_pt3(vec1, vec2, vec3)
    equal(ret, 'out')

    vec3 = latlon2xyz(-pi/4, 5*pi/4)
    ret = arc12_pt3(vec1, vec2, vec3)
    equal(ret, 'out')


    # pt1, pt2
    vec3 = (0,0,1)
    ret = arc12_pt3(vec1, vec2, vec3)
    equal(ret, 'pt1')

    vec3 = latlon2xyz(pi/4, pi/4)
    ret = arc12_pt3(vec1, vec2, vec3)
    equal(ret, 'pt2')


    #---------------------------------------
    # x=0 line
    #---------------------------------------
    xyz1 = latlon2xyz(pi/2-0.5,pi/2)
    xyz2 = latlon2xyz(pi/2-0.2,pi/2)
    xyz3 = latlon2xyz(pi/2-0.1,pi/2)

    ret = arc12_pt3(xyz1, xyz2, xyz3)
    equal(ret, 'out')


    #---------------------------------------
    # left, not straight (??)
    # straight
    #---------------------------------------
    latlons = [(-0.006213200654473781,  4.1626565784079901), 
               (-0.0061246818142171597, 4.1626102660064754),
               (-0.0061246818142039828, 4.1626102660064754)]
    xyz1, xyz2, xyz3 = [latlon2xyz(*latlon) for latlon in latlons]
    ret = arc12_pt3(xyz1, xyz2, xyz3)
    equal(ret, 'out')
    ret = arc12_pt3(xyz2, xyz3, xyz1)
    equal(ret, 'out')
    ret = arc12_pt3(xyz3, xyz1, xyz2)
    equal(ret, 'between')
def test_pt_in_polygon():
    '''
    pt_in_polygon(): out, border, in
    '''

    from sphere import pt_in_polygon
    from convert_coord.cart_ll import latlon2xyz


    # out
    polygon = [latlon2xyz(*ll) for ll in [(pi/2,0), (0,0), (0,pi/4)]]
    point = latlon2xyz(pi/4,2*pi/6)
    ret = pt_in_polygon(polygon, point)
    equal(ret, 'out')


    polygon = [latlon2xyz(*ll) for ll in [(pi/2,0), (0,0), (0,pi/4)]]
    point = latlon2xyz(-pi/6,pi/4)
    ret = pt_in_polygon(polygon, point)
    equal(ret, 'out')


    # border
    polygon = [latlon2xyz(*ll) for ll in [(pi/2,0), (0,0), (0,pi/4)]]
    point = latlon2xyz(pi/4,pi/4)
    ret = pt_in_polygon(polygon, point)
    equal(ret, 'border')


    # in
    polygon = [latlon2xyz(*ll) for ll in [(pi/2,0), (0,0), (0,pi/4)]]
    point = latlon2xyz(pi/4,pi/6)
    ret = pt_in_polygon(polygon, point)
    equal(ret, 'in')




def test_intersect_two_greatcircles():
    '''
    intersect_two_greatcircles(): axis circles, oblique circles, identical
    '''

    from sphere import plane_origin, intersect_two_greatcircles
    from convert_coord.cart_ll import latlon2xyz


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




def test_intersect_two_arcs():
    '''
    intersect_two_arcs(): interset, end point, None
    '''

    from sphere import intersect_two_arcs
    from convert_coord.cart_ll import latlon2xyz, xyz2latlon


    #---------------------------------------
    # intersect
    #---------------------------------------
    xyz1 = latlon2xyz(pi/2,0)
    xyz2 = latlon2xyz(0,pi/4)
    xyz3 = latlon2xyz(0,0)
    xyz4 = latlon2xyz(pi/4,pi/2)
    ret = intersect_two_arcs(xyz1, xyz2, xyz3, xyz4)
    aa_equal(ret, latlon2xyz(0.61547970867038715,pi/4), 15)


    xyz1 = latlon2xyz(pi/4,pi/2)
    xyz2 = latlon2xyz(pi/4,3*pi/2)
    xyz3 = latlon2xyz(pi/4,0)
    xyz4 = latlon2xyz(pi/4,pi)
    ret = intersect_two_arcs(xyz1, xyz2, xyz3, xyz4)
    aa_equal(ret, latlon2xyz(pi/2,0), 15)


    #---------------------------------------
    # end point
    #---------------------------------------
    xyz1 = latlon2xyz(pi/2,0)
    xyz2 = latlon2xyz(pi/4,3*pi/2)
    xyz3 = latlon2xyz(pi/4,0)
    xyz4 = latlon2xyz(pi/4,pi)
    ret = intersect_two_arcs(xyz1, xyz2, xyz3, xyz4)
    equal(ret, 'pt1')


    xyz1 = latlon2xyz(pi/4,pi/2)
    xyz2 = latlon2xyz(pi/2,0)
    xyz3 = latlon2xyz(pi/4,0)
    xyz4 = latlon2xyz(pi/4,pi)
    ret = intersect_two_arcs(xyz1, xyz2, xyz3, xyz4)
    equal(ret, 'pt2')


    xyz1 = latlon2xyz(pi/4,pi/2)
    xyz2 = latlon2xyz(pi/4,3*pi/2)
    xyz3 = latlon2xyz(pi/2,0)
    xyz4 = latlon2xyz(pi/4,pi)
    ret = intersect_two_arcs(xyz1, xyz2, xyz3, xyz4)
    equal(ret, 'pt3')


    xyz1 = latlon2xyz(pi/4,pi/2)
    xyz2 = latlon2xyz(pi/4,3*pi/2)
    xyz3 = latlon2xyz(pi/4,0)
    xyz4 = latlon2xyz(pi/2,0)
    ret = intersect_two_arcs(xyz1, xyz2, xyz3, xyz4)
    equal(ret, 'pt4')


    #---------------------------------------
    # not intersect
    #---------------------------------------
    xyz1 = latlon2xyz(2*pi/6,3*pi/2)
    xyz2 = latlon2xyz(pi/4,3*pi/2)
    xyz3 = latlon2xyz(pi/4,0)
    xyz4 = latlon2xyz(pi/4,pi)
    ret = intersect_two_arcs(xyz1, xyz2, xyz3, xyz4)
    equal(ret, None)




def test_intersect_two_polygons():
    '''
    intersect_two_polygons(): inclusion, partial
    '''

    from sphere import intersect_two_polygons
    from math import pi
    from convert_coord.cart_ll import latlon2xyz


    # inclusion
    ll_poly1 = [(pi/2,0), (0,0), (0,pi/2)]
    ll_poly2 = [(pi/6,pi/6), (pi/6,pi/3), (pi/3,pi/3), (pi/3,pi/6)]
    xyz_poly1 = [latlon2xyz(*ll) for ll in ll_poly1]
    xyz_poly2 = [latlon2xyz(*ll) for ll in ll_poly2]
    ret = intersect_two_polygons(xyz_poly1, xyz_poly2)
    a_equal(ret, xyz_poly2)


    # partial
    ll_poly1 = [(pi/2,0), (0,0), (0,pi/2)]
    ll_poly2 = [(pi/2,0), (0,pi/3), (0,2*pi/3)]
    xyz_poly1 = [latlon2xyz(*ll) for ll in ll_poly1]
    xyz_poly2 = [latlon2xyz(*ll) for ll in ll_poly2]
    ret = intersect_two_polygons(xyz_poly1, xyz_poly2)

    ll_expect = [(pi/2,0), (0,pi/3), (0,pi/2)]
    xyz_expect = [latlon2xyz(*ll) for ll in ll_expect]
    a_equal(ret, xyz_expect)
