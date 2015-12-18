#------------------------------------------------------------------------------
# filename  : sphere.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.17    copy from pygecore_2014.6.23
#             2015.12.18    add distance3, angle3 for Delaunay flipping
#
#
# Description: 
#   Area of a polygon on the sphere
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from math import fsum, sqrt, tan, atan, acos

from util.misc.compare_float import feq




def angle(xyz1, xyz2):
    '''
    angle <AOB
    '''
    cross = np.linalg.norm( np.cross(xyz1, xyz2) ) 
    dot = np.dot(xyz1, xyz2)

    return np.arctan2(cross,dot)




def distance3(xyz1, xyz2):
    x1, y1, z1 = xyz1
    x2, y2, z2 = xyz2

    return sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)




def angle3(xyz1, xyz2, xyz3):
    '''
    angle <ABC
    '''
    d12 = distance3(xyz1, xyz2)
    d23 = distance3(xyz2, xyz3)
    d13 = distance3(xyz1, xyz3)

    return acos( (d12**2 + d23**2 - d13**2)/(2*d12*d23) )




def area_triangle(xyz1, xyz2, xyz3):
    '''
    Area of spherical triangle using Girard theorem
    '''
    a = angle(xyz1, xyz2)
    b = angle(xyz2, xyz3)
    c = angle(xyz3, xyz1)
    s = 0.5*(a+b+c)

    inval = tan(0.5*s) * tan(0.5*(s-a)) * tan(0.5*(s-b)) * tan(0.5*(s-c))
    area = 0 if inval < 0 else 4*atan( sqrt(inval) ) 

    return area




def area_polygon(xyzs):
    '''
    Area from divided spherical triangles
    '''
    area_triangles = list()
    xyz1 = xyzs[0]
    for xyz2, xyz3 in zip(xyzs[1:-1], xyzs[2:]):
        area_tri = area_triangle(xyz1, xyz2, xyz3)
        area_triangles.append(area_tri)

    area = fsum(area_triangles)

    return area




def plane_origin(xyz1, xyz2):
    '''
    return plane parameters (a,b,c) from two points on the sphere
    '''

    x1, y1, z1 = xyz1
    x2, y2, z2 = xyz2
    a, b, c = y1*z2-z1*y2, z1*x2-x1*z2, x1*y2-y1*x2

    return (a, b, c)




def intersect_two_greatcircles((a1, b1, c1), (a2, b2, c2)):
    '''
    Two intersection points of two planes and a sphere
    Input are plane parameters.
    '''

    if feq(a1,a2) and feq(b1,b2) and feq(c1,c2):
        return None, None

    elif feq(a1,0) and feq(a2,0):
        s1, s2 = [(1,0,0), (-1,0,0)]

    elif feq(b1,0) and feq(b2,0):
        s1, s2 = [(0,1,0), (0,-1,0)]

    elif feq(c1,0) and feq(c2,0):
        s1, s2 = [(0,0,1), (0,0,-1)]

    elif feq(c2*b1, c1*b2):
        A = a2*c1-a1*c2
        B = a2*b1-a1*b2
        denom = A*A + B*B

        #if feq(denom,0):
        if denom == 0:
            return None, None
        else:
            Z = 1/np.sqrt(denom)
            s1 = (0, A*Z, -B*Z)
            s2 = (0, -A*Z, B*Z)

    else:
        A = b2*c1-b1*c2
        B = c2*a1-c1*a2
        C = b2*a1-b1*a2
        denom = A*A + B*B + C*C

        #if feq(denom,0):
        if denom == 0:
            return None, None
        else:
            Z = 1/np.sqrt(denom)
            s1 = (A*Z, B*Z, -C*Z)
            s2 = (-A*Z, -B*Z, C*Z)

    return s1, s2
