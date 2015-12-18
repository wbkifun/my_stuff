from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_GreatCircle():
    '''
    GreatCircle: max_angle, pt3
    '''
    from numpy import deg2rad
    from greatcircle import GreatCircle
    from ..convert_coord.cart_ll import xyz2latlon


    #----------------------------------------------
    pt1 = (deg2rad(0),deg2rad(0))
    pt2 = (deg2rad(30),deg2rad(0))

    gc = GreatCircle(pt1, pt2)
    aa_equal(gc.max_angle, np.pi/6, 15)
    aa_equal(gc.latlon3, (np.pi/2,0), 15)


    #----------------------------------------------
    a = 1/np.sqrt(2)
    pt1 = xyz2latlon(1,0,0)
    pt2 = xyz2latlon(a,0.5,0.5)

    gc = GreatCircle(pt1, pt2)
    aa_equal(gc.max_angle, np.pi/4, 15)
    aa_equal(gc.xyz3, (0,a,a), 15)
