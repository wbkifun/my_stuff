#------------------------------------------------------------------------------
# filename  : test_quadrature.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.4   revise
#------------------------------------------------------------------------------
from __future__ import division
import numpy
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_



def test_legendre():
    from quadrature import legendre, deriv_legendre, recursive_L_dL
    x = numpy.arange(-1, 1, 2/100, 'f16')

    for N in xrange(9):
        L = legendre(N, x)
        dL = deriv_legendre(N, x)
        P, dP = recursive_L_dL(N, x)

        aa_equal(L, P, 14)
        aa_equal(dL, dP, 13)



def test_gausslobatto():
    # setup
    gll_pts0 = [-1, -0.830223896278567, -0.468848793470714, 0]
    gll_wts0 = [4.761904761904762E-002, 0.276826047361566, 0.431745381209863, 0.487619047619048]

    # run
    from quadrature import gausslobatto
    gll_pts, gll_wts = gausslobatto(6)

    # verify
    aa_equal(gll_pts[:4], gll_pts0, 15)
    aa_equal(gll_wts[:4], gll_wts0, 15)




def test_gq_integrate_lobatto():
    from quadrature import GQIntegrate

    gqi = GQIntegrate()

    func = lambda x: 2*x*x
    intf = lambda x: 2/3*x**3
    x1, x2 = numpy.random.rand(2)

    ref = intf(x2) - intf(x1)
    aa_equal(ref, gqi.gq_integrate(x1, x2, func), 15)




def test_gq_integrate_legendre():
    from quadrature import GQIntegrate

    gqi = GQIntegrate()

    func = lambda x: 2*x*x
    intf = lambda x: 2/3*x**3
    x1, x2 = numpy.random.rand(2)

    ref = intf(x2) - intf(x1)
    aa_equal(ref, gqi.gq_integrate(x1, x2, func, qtype='legendre'), 15)




def test_gq_integrate_2d():
    from quadrature import GQIntegrate

    gqi = GQIntegrate()

    func = lambda x,y: 2*x*x + y*y
    intf = lambda x,y: 2/3*x**3*y + 1/3*y**3*x
    x1, x2, y1, y2 = numpy.random.rand(4)

    ref = (intf(x2,y2) - intf(x1,y2)) - (intf(x2,y1) - intf(x1,y1)) 
    aa_equal(ref, gqi.gq_integrate_2d(x1, x2, y1, y2, func), 15)
