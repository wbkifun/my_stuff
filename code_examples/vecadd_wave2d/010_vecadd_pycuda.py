#!/usr/bin/env python
#---------------------------------------------------------------------------
# File Name : vecadd_pycuda.py
#
# Author : Ki-Hwan Kim (wbkifun@nate.com)
# 
# Written date :	2011. 6. 22
# Modify date :		
#
# Copyright : GNU GPL
#
# Description : 
# Simple example for pycuda and pyopencl
# Add two vectors
#
# Step 1: PyCUDA
#---------------------------------------------------------------------------

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


# Host(CPU) result
nx = 1024
a = np.random.rand(nx).astype(np.float32)
b = np.random.rand(nx).astype(np.float32)
c = a[:] + b[:]


# GPU result
kernels = """
__global__ void vecadd(int nx, float *a, float *b, float *c) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	
	if( gid < nx ) {
		c[gid] = a[gid] + b[gid];
	}
}
"""
from pycuda.compiler import SourceModule
mod = SourceModule(kernels)
vecadd = mod.get_function('vecadd')

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.mem_alloc(c.nbytes)
c_from_gpu = np.zeros_like(c)

vecadd(np.int32(nx), a_gpu, b_gpu, c_gpu, block=(256,1,1), grid=(nx/256+1,1))
cuda.memcpy_dtoh(c_from_gpu, c_gpu)


# Verify
assert np.linalg.norm(c - c_from_gpu) == 0
