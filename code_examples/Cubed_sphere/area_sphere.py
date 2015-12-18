#------------------------------------------------------------------------------
# filename  : area_sphere.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.17    copy from pygecore_2014.6.23
#
#
# Description: 
#   Area of a polygon on the sphere
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from math import fsum, tan, atan, sqrt




def angle_sphere(xyz1, xyz2):
    '''
    angle <AOB
    '''
    cross = np.linalg.norm( np.cross(xyz1, xyz2) ) 
    dot = np.dot(xyz1, xyz2)

    return np.arctan2(cross,dot)




def area_triangle_sphere(xyz1, xyz2, xyz3):
    '''
    Area of spherical triangle using Girard theorem
    '''
    a = angle_sphere(xyz1, xyz2)
    b = angle_sphere(xyz2, xyz3)
    c = angle_sphere(xyz3, xyz1)
    s = 0.5*(a+b+c)

    inval = tan(0.5*s) * tan(0.5*(s-a)) * tan(0.5*(s-b)) * tan(0.5*(s-c))
    area = 0 if inval < 0 else 4*atan( sqrt(inval) ) 

    return area




def area_polygon_sphere(xyzs):
    '''
    Area from divided spherical triangles
    '''
    area_triangles = list()
    xyz1 = xyzs[0]
    for xyz2, xyz3 in zip(xyzs[1:-1], xyzs[2:]):
        area_triangle = area_triangle_sphere(xyz1, xyz2, xyz3)
        area_triangles.append(area_triangle)

    area = fsum(area_triangles)

    return area
