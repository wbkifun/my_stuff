#------------------------------------------------------------------------------
# filename  : cs_ll.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Development Team, KIAPS
# update    : 2014.3.12 start
#             2015.1.28 add ij2ab
#             2016.8.25 fix the relative import path
#
#
# description: 
#   Convert coordinates between cubed-sphere and latlon
#
# subroutines:
#   ij2ab()
#   xy2ab()
#   ab2xy()
#   latlon2xyp()
#   latlon2abp()
#   xyp2latlon()
#   abp2latlon()
#------------------------------------------------------------------------------

import numpy as np
from numpy import pi, tan, arctan, sqrt

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.extend([current_dpath,dirname(current_dpath)])
from cart_ll import latlon2xyz, xyz2latlon
from cart_cs import xyp2xyz, xyz2xyp
from cart_rotate import xyz_rotate, xyz_rotate_reverse
from misc.quadrature import gausslobatto


RLAT, RLON = 0, 0
#RLAT, RLON = np.deg2rad(38), np.deg2rad(127)      # korea centered




def ij2ab(ne, ngq, ei, ej, gi, gj):
    gq_pts, gq_wts = gausslobatto(ngq-1)
    delta_angles = (gq_pts[:] + 1)*np.pi/(4*ne) 

    alpha = -np.pi/4 + np.pi/(2*ne)*(ei-1) + delta_angles[gi-1]
    beta  = -np.pi/4 + np.pi/(2*ne)*(ej-1) + delta_angles[gj-1]

    #return np.float64(alpha), np.float64(beta)
    return alpha, beta




def xy2ab(x, y, R=1):
    a = R/sqrt(3)

    alpha = arctan(x/a)
    beta = arctan(y/a)

    return alpha, beta




def ab2xy(alpha, beta, R=1):
    a = R/sqrt(3)

    x = a*tan(alpha)
    y = a*tan(beta)

    return x, y




def latlon2xyp(lat, lon, rotate_lat=RLAT, rotate_lon=RLON, R=1):
    '''
    (x,y,panel): gnomonic projection of rotated cubed-sphere coordinates
                 with (rotate_lat, rorate_lon)
    return {panel:(x,y), ...}
    '''

    xyz = latlon2xyz(lat, lon, R)
    xr, yr, zr = xyz_rotate(xyz, rotate_lat, rotate_lon) 
    xyp_dict = xyz2xyp(xr, yr, zr)

    return xyp_dict




def latlon2abp(lat, lon, rotate_lat=RLAT, rotate_lon=RLON, R=1):
    '''
    (alpha,beta,panel): rotated cubed-sphere coordinates 
                        with (rotate_lat, rorate_lon)
    return {panel:(x,y), ...}
    '''

    xyp_dict = latlon2xyp(lat, lon, rotate_lat, rotate_lon, R)

    abp_dict = dict()
    for panel, (x,y) in xyp_dict.items():
        alpha, beta = xy2ab(x, y, R)
        abp_dict[panel] = (alpha, beta)

    return abp_dict




def xyp2latlon(x, y, panel, rotate_lat=RLAT, rotate_lon=RLON, R=1):
    xyz = xyp2xyz(x, y, panel)
    xr, yr, zr = xyz_rotate_reverse(xyz, rotate_lat, rotate_lon)
    lat, lon = xyz2latlon(xr, yr, zr, R)

    '''
    print 'x=%.18f, y=%.18f'%(x,y)
    print 'x=%.18f, y=%.18f, z=%.18f'%xyz
    print 'xr=%.18f, yr=%.18f, zr=%.18f'%(xr,yr,zr)
    print 'lat=%.18f, lon=%.18f'%(lat,lon)
    '''

    return (lat, lon)




def abp2latlon(alpha, beta, panel, rotate_lat=RLAT, rotate_lon=RLON, R=1):
    x, y = ab2xy(alpha, beta, R)
    lat, lon = xyp2latlon(x, y, panel, rotate_lat, rotate_lon, R)

    return (lat, lon)
