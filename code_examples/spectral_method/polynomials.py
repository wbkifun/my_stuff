from __future__ import division
import numpy


TOL = numpy.finfo(numpy.float128).eps
pi = numpy.pi




def legendre_polynomial(k, x):
    '''
    k: degree of polynomial
    x: scalar or array (-1 <= x <= 1)
    return: polynomial and derivative
    '''

    x = numpy.float128(x)

    if k == 0:
        leg = numpy.ones_like(x)
        dleg = numpy.zeros_like(x)

    elif k == 1:
        leg = x
        dleg = numpy.ones_like(x)

    else:
        m1, m2 = x, 1
        dm1, dm2 = 1, 0
        for i in xrange(2,k+1):
            leg = (1/i) * ((2*i-1)*x*m1 - (i-1)*m2)
            dleg = (2*i-1)*m1 + dm2
            m1, m2 = leg, m1
            dm1, dm2 = dleg, dm1
            
    return leg, dleg




def legendre_gauss_nodes_weights(n):
    '''
    return: nodes and weights of Gauss-Legendre Quadrature
    '''

    x = numpy.zeros(n+1, numpy.float128)
    w = numpy.zeros(n+1, numpy.float128)

    if n == 0:
        x, w = 0, 2

    elif n == 1:
        x[0], w[0] = -numpy.sqrt(1/3), 1
        x[1], w[1] = -x[0], w[0]

    else:
        for i in xrange((n+1)//2):
            x[i] = - numpy.cos( (2*i+1)/(2*n+2)*pi )

            while True:
                leg, dleg = legendre_polynomial(n+1, x[i])
                delta = leg/dleg
                x[i] -= delta
                if abs(delta) <= TOL*abs(x[i]): break

            leg, dleg = legendre_polynomial(n+1, x[i])
            x[n-i] = -x[i]
            w[i] = 2/( (1-x[i]**2)*dleg**2 )
            w[n-i] = w[i]

    if n != 0 and numpy.remainder(n,2) == 0:
        leg, dleg = legendre_polynomial(n+1, 0)
        x[n/2] = 0
        w[n/2] = 2/(dleg**2)

    return x, w




def legendre_gauss_lobatto_nodes_weights(n):
    '''
    return: nodes and weights of Gauss-Legendre-Lovatto Quadrature
    '''

    x = numpy.zeros(n+1, numpy.float128)
    w = numpy.zeros(n+1, numpy.float128)

    if n == 1:
        x[0], w[0] = -1, 1
        x[1], w[1] = 1, w[0]

    else:
        x[0], w[0] = -1, 2/( n*(n+1) )
        x[-1], w[-1] = 1, w[0]

        for i in xrange((n+1)//2):
            x[i] = - numpy.cos( (i+1/4)*pi/n - 3/(8*n*pi)/(i+1/4) )

            while True:
                leg_m1, dleg_m1 = legendre_polynomial(n-1, x[i])
                leg_p1, dleg_p1 = legendre_polynomial(n+1, x[i])
                q = leg_p1 - leg_m1
                dq = dleg_p1 - dleg_m1

                delta = q/dq
                x[i] -= delta
                if abs(delta) <= TOL*abs(x[i]): break

            leg, dleg = legendre_polynomial(n, x[i])
            x[n-i] = -x[i]
            w[i] = 2/( n*(n+1)*leg**2 )
            w[n-i] = w[i]

    if numpy.remainder(n,2) == 0:
        leg, dleg = legendre_polynomial(n, 0)
        x[n/2] = 0
        w[n/2] = 2/( n*(n+1)*leg**2 )

    return x, w
