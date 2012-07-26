#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
import pycuda.driver as cuda
import pycuda.autoinit


def vecadd(a, b, c):
	c[:] = a[:] + b[:]


# prepare the kernel
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void vecadd_gpu(int nx, float *a_gpu, float *b_gpu, float *c_gpu) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<nx) c_gpu[i] = a_gpu[i] + b_gpu[i];
}
""")
vecadd_gpu = mod.get_function("vecadd_gpu")


# allocate arrays with initialize
nx = 1000;
a = np.random.randn(nx).astype(np.float32)
b = np.random.randn(nx).astype(np.float32)
c = np.zeros(nx, 'f')
c2 = np.zeros_like(c)

# allocate device arrays with memcpy
a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.mem_alloc(c.nbytes)

# exec
vecadd(a, b, c)
vecadd_gpu(np.int32(nx), a_gpu, b_gpu, c_gpu, block=(256,1,1), grid=(nx/256+1,1))

# copy result and compare
cuda.memcpy_dtoh(c2, c_gpu)
assert la.norm(c2-c) == 0
