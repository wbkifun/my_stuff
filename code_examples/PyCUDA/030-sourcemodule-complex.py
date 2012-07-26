#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

cuda.init()
ctx = cuda.Device(0).make_context()

psi = np.random.randn(5,7).astype(np.complex64) 
psi += 1j
vc = np.zeros(1, dtype=np.complex64)
vc[0] = 0.9 - 0.02j
#print vc * psi

kernels = '''
__constant__ float2 vc[1];

__global__ void vcf(float2 *psi) {
	int tid = threadIdx.x;

	__shared__ float2 spsi[TID_MAX];
	spsi[tid] = psi[tid];

	if ( tid < TID_MAX ) {
		psi[tid].x = vc[0].x * spsi[tid].x - vc[0].y * spsi[tid].y;
		psi[tid].y = vc[0].x * spsi[tid].y + vc[0].y * spsi[tid].x;
		//psi[tid].x = vc[0].x;
		//psi[tid].y = vc[0].y;
	}
}'''.replace('TID_MAX', str(35))
print kernels
mod = SourceModule(kernels)
vcf = mod.get_function('vcf')
vc_const, _ = mod.get_global('vc')
cuda.memcpy_htod(vc_const, vc)

psi_gpu = gpuarray.to_gpu(psi)
vcf(psi_gpu, block=(256,1,1), grid=(1,1))

assert np.linalg.norm(psi_gpu.get() - vc*psi) < 1e-6


ctx.pop()
