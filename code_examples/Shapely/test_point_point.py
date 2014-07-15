from __future__ import division
import numpy as np
from numpy import sqrt
from shapely.geometry import Point
from numpy.testing import assert_equal as assert_eq
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae
from numpy.testing import assert_approx_equal as assert_ape
from nose.tools import ok_




def test_point_point():
    p1 = Point(0,0)
    p2 = Point(1,1)

    ok_(p1.geom_type=='Point', p1.geom_type)
    assert_ae(p1.coords[:][0], (0,0))

    d = p1.distance(p2)
    assert_eq(d, sqrt(2))




def test_point_equal():
    p1 = Point(0,0.001234567890123456786)
    p2 = Point(0,0.001234567890123456987)

    eq = p1.equals(p2)
    ok_(eq==False, eq) 

    aeq = p1.almost_equals(p2,18)
    ok_(aeq==True, aeq) 

    assert_aae(p1.coords[:][0], p2.coords[:][0],18)

    assert_ape(p1.coords[:][0][1], p2.coords[:][0][1],16)
