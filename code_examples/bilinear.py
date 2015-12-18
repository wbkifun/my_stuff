#------------------------------------------------------------------------------
# filename  : bilinear.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Development Team, KIAPS
# update    : 2014.8.19     start
#
#
# description: 
#   Bilinear interpolation at retangular grid
#
# subroutine:
#   bilinear()
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np




def bilinear(x, y, points):
    '''
    Interpolate (x,y) from values associated with four points.
    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear(12, 5.5,
        ...          [(10, 4, 100),
        ...           (10, 6, 150),
        ...           (20, 4, 200),
        ...           (20, 6, 300)])
        165.0
    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)     # order points by x, then by y
    (x1,y1,q11), (_x1,y2,q12), (x2,_y1,q21), (_x2,_y2,q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle') 
    '''
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        print 'x,y', x,y
        print points
        raise ValueError('(x, y) not within the rectangle')
    '''

    
    return (q11 * (x2 - x) * (y2 - y) + 
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) + 
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)



if __name__ == '__main__':
    import doctest
    doctest.testmod()

    print bilinear(1.0000000001,0.55,[(0,0,12),(0,1,13),(1,0,14),(1,1,15)])
