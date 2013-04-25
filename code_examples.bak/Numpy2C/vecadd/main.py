import numpy as np

# import vecop
import imp
program = imp.load_dynamic('vecop', '/home/kifang/Programming/Numpy2C/vecadd/vecop.so')

nx = 4**10

a = np.random.rand(nx).astype(np.float32)
b = np.random.rand(nx).astype(np.float32)
c = np.zeros_like(a)

for i in xrange(10000):
    program.vecadd(a, b, c)

print 'OK!' if np.linalg.norm(c - (a+b)) == 0 else 'Failed!'
