#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy.linalg as la

funcs = """
__global__ void add_gpu(int n, float *a, float *b, float *c) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if( idx<n )
		c[idx] = a[idx]+b[idx];
}

__device__ void add_cpu(int n, float *a, float *b, float *c) {
	int i;
	for( i=0; i<n; i++ )
		c[i] = a[i]+b[i];
}
"""

if __name__ == '__main__':
	nx = 256;

	a = np.random.randn(nx).astype(np.float32)
	b = np.random.randn(nx).astype(np.float32)
	c = np.zeros(nx,'f')
	c2 = np.zeros(nx,'f')

	a_gpu = cuda.to_device(a)
	b_gpu = cuda.to_device(b)
	c_gpu = cuda.to_device(c)

	mod = SourceModule( funcs )
	add_gpu = mod.get_function("add_gpu")
	add_cpu = mod.get_function("add_cpu")

	add_gpu( np.int32(nx), a_gpu, b_gpu, c_gpu, block=(256,1,1), grid=(1,1) )
	cuda.memcpy_dtoh(c2, c_gpu)
	
	assert la.norm(c2 - (a+b)) == 0
