#------------------------------------------------------------------------------
# filename  : great_circle.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2014.4.17     start
#             2016.8.25     fix the relative import path
#
#
# description: 
#   find the great circle from two points on the sphere
#
# Class:
#   GreatCircle
#------------------------------------------------------------------------------

import numpy as np

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.extend([current_dpath,dirname(current_dpath)])
from convert_coord.cart_ll import xyz2latlon, latlon2xyz
from misc.compare_float import feq
from sphere import angle




class GreatCircle(object):
    def __init__(self, latlon1, latlon2):
        self.latlon1 = latlon1
        self.latlon2 = latlon2
        self.xyz1 = latlon2xyz(*latlon1)
        self.xyz2 = latlon2xyz(*latlon2)

        self.max_angle = angle(self.xyz1, self.xyz2)

        # pt3 which is perpendicular with pt1 on the great circle pt1_pt2
        self.xyz3 = self.get_xyz3()
        self.latlon3 = xyz2latlon(*self.xyz3)



    def get_xyz3(self):
        angle = self.max_angle
        x1, y1, z1 = self.xyz1
        x2, y2, z2 = self.xyz2
        
        c_angle, s_angle = np.cos(angle), np.sin(angle)
        if feq(s_angle,0):
            x3, y3, z3 = x2, y2, z2
        else:
            x3 = (x2 - x1*c_angle)/s_angle
            y3 = (y2 - y1*c_angle)/s_angle
            z3 = (z2 - z1*c_angle)/s_angle

        return (x3,y3,z3)



    def phi2xyz(self, phi):
        x1, y1, z1 = self.xyz1
        x3, y3, z3 = self.xyz3

        c_phi, s_phi = np.cos(phi), np.sin(phi)
        x = x1*c_phi + x3*s_phi
        y = y1*c_phi + y3*s_phi
        z = z1*c_phi + z3*s_phi

        return (x, y, z)



    def phi2latlon(self, phi):
        return xyz2latlon( *self.phi2xyz(phi) )
