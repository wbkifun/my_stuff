#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


bs = 16		# block size
nx, ny = bs, bs
a = np.random.rand(nx,ny).astype(np.float32)


kernels = """
__global__ void BlockTranspose(float* A_elements, int A_width) {
	__shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE];

	int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

	blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
	__syncthreads();
	A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
"""
from pycuda.compiler import SourceModule
mod = SourceModule( kernels.replace('BLOCK_SIZE', str(bs)) )
BlockTranspose = mod.get_function('BlockTranspose')

a_gpu = cuda.to_device(a)
a_from_gpu = np.zeros_like(a)

BlockTranspose(a_gpu, np.int32(nx), block=(bs,bs,1), grid=(nx/bs,ny/bs))
cuda.memcpy_dtoh(a_from_gpu, a_gpu)

assert np.linalg.norm(a.T - a_from_gpu) == 0
