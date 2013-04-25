import numpy as np
from ctypes import c_int


nx = 2**20

a = np.random.rand(nx).astype(np.float32)
b = np.random.rand(nx).astype(np.float32)
c = np.zeros_like(a)

program = np.ctypeslib.load_library('vecadd', '.')
carg = np.ctypeslib.ndpointer(a.dtype, ndim=a.ndim, shape=a.shape, flags='C_CONTIGUOUS, ALIGNED')
program.vecadd.argtypes = [c_int, carg, carg, carg]
program.vecadd.resype = None

program.vecadd(np.int32(nx), a, b, c)

print 'OK!' if np.linalg.norm(c - (a+b)) == 0 else 'Failed!'
