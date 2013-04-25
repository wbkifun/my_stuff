#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np


nx, ny = 1024, 1024
#nx, ny = 5, 9
a = np.zeros((nx, ny), dtype=np.complex64)
b = np.arange(nx, dtype=np.complex64)
#print 'reference'
#print b
a[:] = b[:,np.newaxis]
#print a

#print 'gpuarray:'
a_gpu = gpuarray.zeros((nx, ny), dtype=np.complex64)
b_gpu = gpuarray.to_gpu(b)

kernels = '''
__global__ void kern(float2 *a, float2 *b) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid < NXY ) {
		a[tid].x = b[tid/NY].x;
		a[tid].y = b[tid/NY].y;
	}
}'''.replace('NXY',str(nx*ny)).replace('NX',str(nx)).replace('NY',str(ny))
#print kernels
mod = SourceModule(kernels)
kern = mod.get_function("kern")
kern(a_gpu, b_gpu, block=(256,1,1), grid=(nx*ny/256+1,1))
#print a_gpu.get()
print np.linalg.norm(a - a_gpu.get())
