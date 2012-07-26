#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np

kernels = '''
__global__ void cumul(float2 *a, float2 *b, float2 *c) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while ( tid < NN ) {
		c[tid].x = a[tid].x * b[tid].x - a[tid].y * b[tid].y;
		c[tid].y = a[tid].x * b[tid].y + a[tid].y * b[tid].x;
		tid += gridDim.x * blockDim.x;
	}
}'''

nx, ny = 110, 130

a = np.random.randn(nx,ny).astype(np.complex64)
b = np.random.randn(nx,ny).astype(np.complex64)
c = np.zeros_like(a)

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.mem_alloc(c.nbytes)

# use the normal kernel
from pycuda.compiler import SourceModule
mod = SourceModule(kernels.replace('NN',str(nx*ny)))
cumul = mod.get_function("cumul")
cumul(a_gpu, b_gpu, c_gpu, block=(256,1,1), grid=(2,1))
cuda.memcpy_dtoh(c, c_gpu)
assert np.linalg.norm(c - a*b) < 1e-6

# use the gpuarray
a_ga = gpuarray.to_gpu(a)
b_ga = gpuarray.to_gpu(b)
assert np.linalg.norm((a_ga*b_ga).get() - a*b) < 1e-6

# arguments
cumul(a_ga, b_ga, c_gpu, block=(256,1,1), grid=(2,1))
cuda.memcpy_dtoh(c, c_gpu)
assert np.linalg.norm(c - a*b) < 1e-6

# sub-area memcpy from gpuarray
a_sub = np.zeros(100, dtype=np.complex64)
a_sub[:] = cuda.from_device(int(a_ga.gpudata) + 900*np.nbytes['complex64'], (100,), 'complex64')
print a_sub

d_g = gpuarray.zeros((10,13), dtype=np.complex64)
print cuda.from_device(int(d_g.gpudata) + 100*np.nbytes['complex64'], (30,), 'complex64')
