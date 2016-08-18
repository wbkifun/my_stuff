from __future__ import division
import numpy as np
from numpy.testing import assert_array_equal as a_equal


'''
y = a*(x[0]+x[1])
'''

n = 100

a = np.random.rand()
x = np.random.rand(n,2)
y = np.zeros(n)

y[:] = a*(x[:,0] + x[:,1])

from axpx_f import axpx
y2 = np.zeros_like(y)
axpx(n, a, x.ravel(), y2)

a_equal(y, y2)
