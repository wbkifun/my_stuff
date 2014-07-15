from __future__ import division
import numpy
from numpy import pi, deg2rad, sin, cos
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae



def lonlat2xyz(lon, lat):
    x = cos(lat)*cos(lon)
    y = cos(lat)*sin(lon)
    z = sin(lat)

    return (x,y,z)



def rotate_z(lon0, x, y, z):
    x2 = cos(lon0)*x - sin(lon0)*y
    y2 = sin(lon0)*x + cos(lon0)*y
    z2 = z

    return (x2,y2,z2)



def rotate_y(lat0, x, y, z):
    x2 = cos(lat0)*x + sin(lat0)*z
    y2 = y
    z2 = -sin(lat0)*x + cos(lat0)*z

    return (x2,y2,z2)



def rotate(lon0, lat0, x, y, z):
    x2 = cos(lat0)*cos(lon0)*x - cos(lat0)*sin(lon0)*y + sin(lat0)*z
    y2 = sin(lon0)*x + cos(lon0)*y
    z2 = -sin(lat0)*cos(lon0)*x + sin(lat0)*sin(lon0)*y + cos(lat0)*z

    x3, y3, z3 = rotate_z(lon0, x, y, z)
    x4, y4, z4 = rotate_y(lat0, x3, y3, z3)

    assert_aae([x2,y2,z2], [x4,y4,z4], 15)

    return (x2,y2,z2)




if __name__ == '__main__':
    lon1 = deg2rad(127.5)
    lat1 = deg2rad(38)

    x1, y1, z1 = lonlat2xyz(lon1, lat1)
    x2, y2, z2 = rotate(-lon1, lat1, x1, y1, z1)
    #x2, y2, z2 = rotate_z(pi/2, 0, 1, 2.7)

    print x2, y2, z2
