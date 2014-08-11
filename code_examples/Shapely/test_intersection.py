from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal

from shapely.geometry import Polygon, LineString




def test_polygon_line():
    '''
    intersection between polygon and line
    '''
    #-------------------------------------------------
    # inclusion
    #-------------------------------------------------
    poly = Polygon([(0,0),(0,1),(1,1),(1,0)])
    line = LineString([(0,0), (0.5,0.5)])

    iline = poly.intersection(line)
    equal(np.sqrt(2)*0.5, iline.length)
    a_equal(line, np.array(iline.coords))


    #-------------------------------------------------
    # partially
    #-------------------------------------------------
    poly = Polygon([(0,0),(0,1),(1,1),(1,0)])
    line = LineString([(0.5,0.5),(1.5,1.5)])

    iline = poly.intersection(line)
    equal(np.sqrt(2)*0.5, iline.length)
    a_equal(LineString([(0.5,0.5),(1,1)]), np.array(iline.coords))


    #-------------------------------------------------
    # not intersection
    #-------------------------------------------------
    poly = Polygon([(0,0),(0,1),(1,1),(1,0)])
    line = LineString([(1,1),(2,2)])

    iline = poly.intersection(line)
    equal(0, iline.length)




def test_line_line():
    '''
    intersection between line and line
    '''
    #-------------------------------------------------
    # intersection
    #-------------------------------------------------
    line1 = LineString([(0,0), (1,1)])
    line2 = LineString([(1,0), (0,1)])

    ist = line1.intersection(line2)
    equal(ist.geom_type, 'Point')
    a_equal([0.5,0.5], np.array(ist.coords)[0])



    #-------------------------------------------------
    # parallel
    # line intersection
    #-------------------------------------------------
    line1 = LineString([(0,0), (1,1)])
    line2 = LineString([(-1,-1), (0.5,0.5)])

    ist = line1.intersection(line2)
    equal(ist.geom_type, 'LineString')
    a_equal([(0,0),(0.5,0.5)], np.array(ist.coords))



    #-------------------------------------------------
    # parallel
    # not intersection
    #-------------------------------------------------
    line1 = LineString([(0,0), (1,1)])
    line2 = LineString([(0,-1), (1,0)])

    ist = line1.intersection(line2)
    equal(True, ist.is_empty)



    #-------------------------------------------------
    # not intersection
    #-------------------------------------------------
    line1 = LineString([(0,0), (1,1)])
    line2 = LineString([(3,0), (0,3)])

    ist = line1.intersection(line2)
    equal(True, ist.is_empty)
