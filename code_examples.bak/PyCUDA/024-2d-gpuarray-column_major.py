#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np


nx, ny = 5, 9
a = np.zeros((nx, ny), dtype=np.complex64)
for i in xrange(nx):
	for j in xrange(ny):
		a[i,j] = i + 1j*j
print 'Numpy default: Row-major'
print a

print 'gpuarray:'
a_gpu = gpuarray.zeros((nx, ny), dtype=np.complex64)
print a_gpu.get()


kernels = '''
__global__ void setij(float2 *a) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;

	if ( tid < NXY ) {
		i = tid / NY;
		j = tid % NY;
		a[tid].x = i;
		a[tid].y = j;
	}
}'''.replace('NXY',str(nx*ny)).replace('NX',str(nx)).replace('NY',str(ny))
print kernels
mod = SourceModule(kernels)
setij = mod.get_function("setij")
setij(a_gpu, block=(256,1,1), grid=(1,1))
print a_gpu.get()
