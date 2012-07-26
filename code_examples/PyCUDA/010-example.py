#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

kernels = '''
__global__ void cumul(int nx, float *a, float *b, float *c) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < nx) {
		c[tid] = a[tid] * b[tid];
		tid += gridDim.x * blockDim.x;
	}
}'''

from pycuda.compiler import SourceModule
mod = SourceModule(kernels)
cumul = mod.get_function("cumul")

nx = 1000
a = np.random.randn(nx).astype(np.float32)
b = np.random.randn(nx).astype(np.float32)
c = np.zeros_like(a)

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.mem_alloc(c.nbytes)

cumul(np.int32(nx), a_gpu, b_gpu, c_gpu, block=(256,1,1), grid=(2,1))
cuda.memcpy_dtoh(c, c_gpu)

assert np.linalg.norm(c - a*b) < 1e-6
