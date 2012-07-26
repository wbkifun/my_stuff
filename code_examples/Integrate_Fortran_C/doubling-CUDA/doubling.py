#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

kernels="""
__global__ void doubling(float *a) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
	a[i] = 2*a[i];
}
"""
from pycuda.compiler import SourceModule
mod = SourceModule(kernels)
doubling = mod.get_function("doubling")

a = np.ones(10, 'f')
a_gpu = cuda.to_device(a)

print 'before', a
doubling(a_gpu, block=(10,1,1), grid=(1,1))
cuda.memcpy_dtoh(a, a_gpu)
print 'after', a
