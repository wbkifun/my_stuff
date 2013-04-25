#!/usr/bin/env python

# Creating arrays
import numpy as np
a = np.ones(10, np.float32)
b = np.arange(-3,2,0.5)
c = np.array([-2, -1.1, 5, 9], np.float64)
d = np.zeros((3,4), np.complex64)

print a, '\n', b, '\n', c, '\n', d


# Array indexing
a = np.arange(-1, 1.01, 0.4)
print a
a[2:4] = -1
a[-1] = a[0]
print a

a.shape = (2,3)
print a
a[:,:] = 0.1
print a

# Array computations
a = np.ones((2,3))
b = 3*a[:,:] - 1
print b
b[:,-1] *= 3
print b
c = a**2 + np.sin(b[:,:]) 
print c

# Slicing
n = 100
beta = 0.5
u = np.ones(n,'f')
u_new = np.ones_like(u)
# plain Python
for i in xrange(1,n-1,1):
	u_new[i] = beta*u[i-1] + (1-2*beta)*u[i] + beta*u[i+1]

# slicing
u[1:-1] = beta*u[:-2] + (1-2*beta)*u[1:-1] + beta*u[2:]

import numpy.linalg as la
assert la.norm(u_new - u) == 0

