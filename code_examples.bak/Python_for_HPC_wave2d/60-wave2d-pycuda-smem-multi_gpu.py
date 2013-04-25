#!/usr/bin/env python
#------------------------------------------------------------------------------
# File Name : wave2d-pycuda.py
#
# Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
# 
# Written date :	2010. 12. 16
# Modify date :		
#
# Copyright : GNU GPL
#
# Description : 
# Simulation for the 2-dimensional wave equations with simple FD (Finite-Difference) scheme
#
# These are educational codes to study python programming for high performance computing.
# Step 1: Using numpy arrays
# Step 2: Convert the performance hotspot to the C function
# Step 2-1: C code Optimization - SIMD vectorize using SSE intrinsics
# Step 2-2: C code Optimization - Utilize multi-core using OpenMP
# Step 3: Extend to MPI
# Step 4: Utilize the GPU using PyCUDA
# Step 5: CUDA Optimization - use shared memory
# Step 6: Extend to Multi-GPU using MPI
#------------------------------------------------------------------------------

import numpy as np
import sys
from matplotlib.pyplot import *
from datetime import datetime
import pycuda.driver as cuda
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()	# Assume size as 2

import pycuda.autoinit


kernels ="""
#define Dx 256

__global__ void update(int nx, int ny, float *c, float *f, float *g) {
	int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;
	int i = idx/ny;

	__shared__ float sm[Dx+2];
	float *s = &sm[1];
	s[tx] = g[idx];
	if( tx==0 && i>0 ) s[tx-1] = g[idx-1];
	if( tx==Dx-1 && i<nx-1 ) s[tx+1] = g[idx+1];
	__syncthreads();
	
	if( i>0 && i<nx-1 )
		f[idx] = c[idx]*(g[idx+ny] + g[idx-ny] + s[tx+1] + s[tx-1] - 4*s[tx]) + 2*s[tx] - f[idx];
}

__global__ void update_src(int nx, int ny, float tn, float *f) {
	f[400*ny+500] += sin(0.1*tn);
}

"""
from pycuda.compiler import SourceModule
mod = SourceModule(kernels)
update = mod.get_function("update")
update_src = mod.get_function("update_src")


nx, ny = 1000, 1000
tmax = 200
c = np.ones((nx,ny),'f')*0.25
f = np.zeros_like(c)
c_gpu = cuda.to_device(c)
f_gpu = cuda.to_device(f)
g_gpu = cuda.to_device(f)

# To plot using matplotlib
ion()
imsh = imshow(np.ones_like(c).T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.2)
colorbar()

# To measure the execution time
t1 = datetime.now()

# Main loop for time evolution
for tn in range(1,tmax+1):
	update_src(np.int32(nx), np.int32(ny), np.float32(tn), g_gpu, block=(1,1,1), grid=(1,1))
	update(np.int32(nx), np.int32(ny), c_gpu, f_gpu, g_gpu, block=(256,1,1), grid=(nx*ny/256+1,1))
	update(np.int32(nx), np.int32(ny), c_gpu, g_gpu, f_gpu, block=(256,1,1), grid=(nx*ny/256+1,1))

	if tn%10 == 0:
		print "tstep =\t%d/%d (%d %%)\r" % (tn, tmax, float(tn)/tmax*100),
		sys.stdout.flush()
		cuda.memcpy_dtoh(f,f_gpu)
		imsh.set_array( np.sqrt(f.T**2) )
		draw()
		#savefig('./png/%.5d.png' % tn) 

print '\n', datetime.now() - t1
