#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np


#nx, ny = 5, 9
nx, ny = 2048, 1024
dx = 0.01
dt = 0.0001
lcx = np.zeros(nx, dtype=np.complex64)
kx = (np.fft.fftfreq(nx, dx) * 2 * np.pi)
lcx[:] = np.exp(- 0.5j * kx**2 * dt)

a = np.zeros((nx, ny), dtype=np.complex64)
#b = np.arange(nx, dtype=np.complex64)
b = lcx
print 'b\n', b
a[:] = b[:,np.newaxis]
print 'a\n', a

a_gpu = gpuarray.zeros((nx, ny), dtype=np.complex64)
b_gpu = gpuarray.to_gpu(b)

print b_gpu.dtype, b_gpu.shape

kernels = '''
__global__ void kern(float2 *a, float2 *b) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid < NXY ) {
		a[tid] = b[tid/NY];
	}
}'''.replace('NXY',str(nx*ny)).replace('NX',str(nx)).replace('NY',str(ny))
print kernels
mod = SourceModule(kernels)
kern = mod.get_function("kern")
kern(a_gpu, b_gpu, block=(256,1,1), grid=(nx*ny/256+1,1))
print 'a_gpu\n', a_gpu.get()
print np.linalg.norm(a - a_gpu.get())
print np.linalg.norm(b - a_gpu.get()[:,0])
