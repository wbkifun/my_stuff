from __future__ import division
from numpy.testing import assert_equal as ae
from numpy.testing import assert_array_almost_equal as aaae
import numpy



def legendre_ref(k, x):
    x = numpy.float128(x)

    if k == 0:
        leg = numpy.ones_like(x)
        dleg = numpy.zeros_like(x)

    elif k == 1:
        leg = x
        dleg = numpy.ones_like(x)

    elif k == 2:
        leg = 1/2*(3*x**2 - 1)
        dleg = 3*x

    elif k == 3:
        leg = 1/2*(5*x**3 - 3*x)
        dleg = 1/2*(15*x**2 - 3)

    elif k == 4:
        leg = 1/8*(35*x**4 - 30*x**2 + 3)
        dleg = 1/2*(35*x**3 - 15*x)

    elif k == 5:
        leg = 1/8*(63*x**5 - 70*x**3 + 15*x)
        dleg = 1/8*(315*x**4 - 210*x**2 + 15)

    elif k == 6:
        leg = 1/16*(231*x**6 - 315*x**4 + 105*x**2 - 5)
        dleg = 1/8*(693*x**5 - 630*x**3 + 105*x)

    else:
        raise ValueError, 'allow k: 0 <= k <= 6,  input k: %d' % k

    return leg, dleg




def test_legendre_polynomial():
    from polynomials import legendre_polynomial

    # scalar
    x = numpy.random.uniform(-1,1)
    for k in xrange(7):
        leg_ref, dleg_ref = legendre_ref(k, x)
        leg, dleg = legendre_polynomial(k, x)
        aaae(leg_ref, leg, 15, 'degree %d, pol' % k)
        aaae(dleg_ref, dleg, 15, 'degree %d, deriv' % k)

    # array
    x = numpy.linspace(-1,1,100)
    for k in xrange(7):
        leg_ref, dleg_ref = legendre_ref(k, x)
        leg, dleg = legendre_polynomial(k, x)
        aaae(leg_ref, leg, 15, 'degree %d, pol' % k)
        aaae(dleg_ref, dleg, 15, 'degree %d, deriv' % k)




def test_legendre_gauss_nodes_weights():
    from polynomials import legendre_gauss_nodes_weights as gl

    # n = 6 
    x_ref = numpy.float128( \
            [-0.949107912342759, -0.741531185599395, -0.405845151377397, 0])
    w_ref = numpy.float128( \
            [0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469])
    x, w = gl(6)
    aaae(x_ref, x[:4], 15, 'n=6, x')
    aaae(w_ref, w[:4], 15, 'n=6, w')




def test_legendre_gauss_lobatto_nodes_weights():
    from polynomials import legendre_gauss_lobatto_nodes_weights as gll

    # n = 6 
    x_ref = numpy.float128([-1, -0.830223896278567, -0.468848793470714, 0])
    w_ref = numpy.float128( \
            [0.04761904761904762, 0.276826047361566, 0.431745381209863, 0.487619047619048])
    x, w = gll(6)
    aaae(x_ref, x[:4], 15, 'n=6, x')
    aaae(w_ref, w[:4], 15, 'n=6, w')




if __name__ == '__main__':
    pass
