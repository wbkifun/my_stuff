#!/usr/bin/env python

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np


nx = 2
dx = 0.01
dt = 0.0001
kx = (np.fft.fftfreq(nx, dx) * 2 * np.pi)
lcx = (np.exp(- 0.5j * kx**2 * dt)).astype(np.complex64)

#b = np.arange(nx, dtype=np.complex64)
b = lcx
print 'b\n', b
print b.dtype

a = np.zeros(nx, dtype=np.complex64)
a[:] = b[:]
print 'a\n', a

a_gpu = gpuarray.zeros(nx, dtype=np.complex64)
b_gpu = gpuarray.to_gpu(b)
print 'b_gpu\n', b_gpu.get()

kernels = '''
__global__ void kern1(float2 *a, float2 *b) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid < NX ) {
		a[tid] = b[tid];
		//a[tid].x = b[tid].x;
		//a[tid].y = b[tid].y;
	}
}

__global__ void kern2(float *a, float *b) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid < 2 * NX ) {
		a[tid] = b[tid];
	}
}

/*
__global__ void cmul(float *a, float *b) {
	int tx = threadIdx.x;
	int tid = blockIdx.x * blockDim.x + tx;
	int real = 2 * tx;
	int imag = 2 * tx + 1;

	__shared__ float sa[512];
	__shared__ float sb[512];
	__shared__ float sc[512];

	if ( tid < NX ) {
		sa[tx] = a[tid];
		sb[tx] = b[tid];
		sa[tx+256] = a[tid+256];
		sb[tx+256] = b[tid+256];
		__syncthreads();

		sc[real] = sa[real] * sb[real] - sa[imag] * sb[imag];
		sc[imag] = sa[real] * sb[imag] + sa[imag] * sb[real];

		a[tid] = sc[tx];
		a[tid+256] = sc[tx+256];
	}
}
*/'''.replace('NX',str(nx))
#print kernels
mod = SourceModule(kernels)
kern1 = mod.get_function("kern1")
kern1(a_gpu, b_gpu, block=(256,1,1), grid=(nx/256+1,1))
print 'a_gpu\n', a_gpu.get()
print np.linalg.norm(a - a_gpu.get())

kern2 = mod.get_function("kern2")
kern2(a_gpu, b_gpu, block=(256,1,1), grid=(nx/256+1,1))
print 'a_gpu\n', a_gpu.get()
print np.linalg.norm(a - a_gpu.get())

print 'b_gpu\n', b_gpu.get()
