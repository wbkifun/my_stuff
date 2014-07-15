from __future__ import division
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_approx_equal
from numpy.testing import assert_array_almost_equal_nulp
from numpy.testing import assert_array_max_ulp




a = np.array( (0,0.0001234567890123456789) )
b = np.array( (0,0.0001234567890123456987) )


print '\nassert_array_equal'
try:
    print assert_array_equal(a,b)
    print True
except Exception, err:
    print False


print '\nassert_array_almost_equal'
for digit in xrange(20,0,-1):
    try:
        assert_array_almost_equal(a,b,digit)
        print 'digit= %d'%digit
        break
    except Exception, err:
        continue


print '\nassert_approx_equal'
for digit in xrange(20,0,-1):
    try:
        assert_approx_equal(a[1],b[1],digit)
        print 'digit= %d'%digit
        break
    except Exception, err:
        continue


print '\nassert_array_almost_nulp'
for nulp in xrange(1,20):
    try:
        assert_array_almost_equal_nulp(a,b,nulp)
        print 'nulp= %d'%nulp
        break
    except Exception, err:
        print err
        continue


print '\nassert_array_max_ulp'
for maxulp in xrange(20,0,-1):
    try:
        res = assert_array_max_ulp(a,b,maxulp)
        print res
        print 'maxulp= %d'%maxulp
        break
    except Exception, err:
        print err
        continue
