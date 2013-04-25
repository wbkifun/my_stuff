from __future__ import division
import numpy
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae


TOL = numpy.finfo(numpy.float64).eps



#------------------------------------------------------------------------------
# Setup 
# 4x4 matrix
A = numpy.array([[10,-1,2,0], [-1,11,-1,3], [2,-1,10,-1], [0,3,-1,8]])
b = numpy.array([6,25,-11,15])
x_exact = (1,2,-1,1)
assert_ae(numpy.dot(A, x_exact), b)



#------------------------------------------------------------------------------
# numpy solve
x = numpy.linalg.solve(A, b)
assert_ae(x, x_exact)



#==============================================================================
# Gauss-Seidel method
#==============================================================================
# naive implementation
x = numpy.zeros(4)

for i in xrange(20):
    x[0]=(b[0]-(A[0,1]*x[1]+A[0,2]*x[2]+A[0,3]*x[3])                                      )/A[0,0]
    x[1]=(b[1]-(            A[1,2]*x[2]+A[1,3]*x[3])-(A[1,0]*x[0]                        ))/A[1,1]
    x[2]=(b[2]-(                        A[2,3]*x[3])-(A[2,0]*x[0]+A[2,1]*x[1]            ))/A[2,2]
    x[3]=(b[3]                                      -(A[3,0]*x[0]+A[3,1]*x[1]+A[3,2]*x[2]))/A[3,3]

assert_aae(x, x_exact, 15)



#------------------------------------------------------------------------------
# function implementation

def gauss_seidel(A, b, max_iter=1000, verbose=False):
    n = b.size
    x = numpy.zeros_like(b, 'f8')

    for step in xrange(max_iter):
        err = 0
        for i in xrange(n):
            x0 = x[i]

            ux = 0
            for j in xrange(i+1,n):
                ux += A[i,j]*x[j]

            lx = 0
            for j in xrange(i):
                lx += A[i,j]*x[j]

            x[i] = (b[i] - ux - lx)/A[i,i]

            err += (x[i] - x0)**2
        if numpy.sqrt(err) <= TOL: break

    if verbose:
        return x, step
    else:
        return x


x, step = gauss_seidel(A, b, verbose=True)
print 'step=', step
assert_aae(x, x_exact, 15)
