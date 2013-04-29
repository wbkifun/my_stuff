from __future__ import division
import numpy
from numpy.testing import assert_array_equal as assert_ae


nx = 4
a = numpy.arange(nx*nx).reshape((nx,nx), order='F') + 1

print a


for j in xrange(nx):
    for i in xrange(nx):
        if i != 0 and j != 0: 
            a[i,j] += a[i-1,j] - a[i,j-1]
        elif i != 0 and j == 0:
            a[i,j] += a[i-1,j]
        elif i == 0 and j != 0:
            a[i,j] -= a[i,j-1]
        #print '\n(%d,%d)' % (i,j)
        #print a

print '\n',a


a0 = a.copy()


