from __future__ import division
import numpy
from numpy.testing import assert_array_equal as assert_ae



# 2x2
A = numpy.array([[3,1], [1,2]])
b = numpy.array([9,8])
x = numpy.linalg.solve(A, b)

print x
assert_ae(x, (2,3))
assert_ae(numpy.dot(A, x), b)



# 4x4
A = numpy.array([[10,-1,2,0], [-1,11,-1,3], [2,-1,10,-1], [0,3,-1,8]])
b = numpy.array([6,25,-11,15])
x = numpy.linalg.solve(A, b)

print x
assert_ae(x, (1,2,-1,1))
assert_ae(numpy.dot(A, x), b)
