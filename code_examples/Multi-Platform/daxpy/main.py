from __future__ import division
import numpy as np
from numpy.testing import assert_array_equal as a_equal


'''
y = a*x + y
'''

n = 100

a = np.random.rand()
x = np.random.rand(n)
y = np.random.rand(n)

y2 = a*x + y

from tmpAVJD import daxpy
y3 = y[:]
daxpy(n, a, x, y3)

a_equal(y2, y3)
