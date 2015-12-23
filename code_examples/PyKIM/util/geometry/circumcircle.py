#------------------------------------------------------------------------------
# filename  : circumcircle.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2014.4.25     start
#             2014.6.12     circum_center_radius()
#             2014.6.30     circum_center_radius_plane()
#             2015.12.18    latlon argumensts -> xyz
#
#
# description: 
#   Find a center and radius of a circumcircle from three points
#
# subroutine:
#   circum_center_radius()
#   circum_center_radius_plane()
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy import sqrt, fabs, sin, cos
from numpy.linalg import det

from util.convert_coord.cart_ll import latlon2xyz, xyz2latlon
from util.misc.compare_float import fne, feq
from sphere import intersect_two_greatcircles, angle




def circum_center_radius(xyz1, xyz2, xyz3):
    x1, y1, z1 = xyz1
    x2, y2, z2 = xyz2
    x3, y3, z3 = xyz3


    # bisector plane of p1 and p2
    mx, my, mz = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2 
    cx, cy, cz = y1*z2-z1*y2, z1*x2-x1*z2, x1*y2-y1*x2
    a1, b1, c1 = my*cz-mz*cy, mz*cx-mx*cz, mx*cy-my*cx      # plane parameters


    # bisector plane of p2 and p3
    mx, my, mz = (x2+x3)/2, (y2+y3)/2, (z2+z3)/2 
    cx, cy, cz = y2*z3-z2*y3, z2*x3-x2*z3, x2*y3-y2*x3
    a2, b2, c2 = my*cz-mz*cy, mz*cx-mx*cz, mx*cy-my*cx      # plane parameters


    # intersection points
    cc1, cc2 = intersect_two_greatcircles((a1,b1,c1), (a2,b2,c2))


    # circumcircle point
    mxyz = (x1+x2+x3)/2, (x2+y2+y3)/2, (x3+z2+z3)/2 
    d1 = angle(cc1, mxyz)
    d2 = angle(cc2, mxyz)

    cc = cc1 if d1 < d2 else cc2
    dist = angle(cc,xyz1)


    return cc, dist




def circum_center_radius_plane(xy1, xy2, xy3):
    x1, y1 = xy1
    x2, y2 = xy2
    x3, y3 = xy3
    xxyy1 = x1**2 + y1**2
    xxyy2 = x2**2 + y2**2
    xxyy3 = x3**2 + y3**2

    a =    det( np.array([[x1,y1,1],
                          [x2,y2,1],
                          [x3,y3,1]]) )

    if fne(a,0):
        bx = - det( np.array([[xxyy1,y1,1],
                              [xxyy2,y2,1],
                              [xxyy3,y3,1]]) )

        by =   det( np.array([[xxyy1,x1,1],
                              [xxyy2,x2,1],
                              [xxyy3,x3,1]]) )

        c = -  det( np.array([[xxyy1,x1,y1],
                              [xxyy2,x2,y2],
                              [xxyy3,x3,y3]]) )

        center = (-bx/(2*a), -by/(2*a))
        radius = sqrt(bx**2+by**2-4*a*c)/(2*fabs(a))
        return center, radius

    else:
        return None, None




class Circumcircle(object):
    def __init__(self, center_latlon, latlon, R=1):
        center_xyz = np.array( latlon2xyz(*center_latlon) )
        xyz = np.array( latlon2xyz(*latlon) )

        d = np.linalg.norm(center_xyz-xyz)
        pc = (R-0.5*d*d/R)*center_xyz       # plane_center_xyz 
        v1 = xyz - pc
        nv1 = v1/np.linalg.norm(v1)
        v2 = np.cross(pc, v1)
        nv2 = v2/np.linalg.norm(v2)


        self.r = np.linalg.norm(pc-xyz)
        self.pc = pc
        self.nv1 = nv1
        self.nv2 = nv2



    def phi2latlon(self, phi):
        r, pc, nv1, nv2 = self.r, self.pc, self.nv1, self.nv2

        xyz = pc + r*cos(phi)*nv1 + r*sin(phi)*nv2
        return xyz2latlon(*xyz)
