#!/usr/bin/env python
#---------------------------------------------------------------------------
# File Name : wave2d_pycuda.py
#
# Author : Ki-Hwan Kim (wbkifun@nate.com)
# 
# Written date :	2010. 6. 17
# Modify date :		
#
# Copyright : GNU GPL
#
# Description : 
# Simulation for the 2-dimensional wave equations with simple FD (Finite-Difference) scheme
#
# These are educational codes to study python programming for high performance computing.
# Step 1: Using numpy arrays
# Step 2: Utilize the GPU using PyCUDA
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit


kernels ="""
__global__ void update(int nx, int ny, float *c, float *f, float *g) {
	int gid = blockIdx.x*blockDim.x + threadIdx.x;
	int i = gid/ny;
	int j = gid%ny;
	
	if( i>0 && i<nx-1 && j>0 && j<ny-1 )
		f[gid] = c[gid]*(g[gid+ny] + g[gid-ny] + g[gid+1] + g[gid-1] - 4*g[gid]) + 2*g[gid] - f[gid];
}


__global__ void update_src(int nx, int ny, float tn, float *f) {
	f[400*ny+300] += sin(0.1*tn);
}
"""
from pycuda.compiler import SourceModule
mod = SourceModule(kernels)
update = mod.get_function("update")
update_src = mod.get_function("update_src")


nx, ny = 1000, 800
tmax = 500
f = np.zeros((nx,ny), dtype=np.float32)
c = np.ones_like(f) * 0.25
f_gpu = cuda.to_device(f)
g_gpu = cuda.to_device(f)
c_gpu = cuda.to_device(c)

# To plot using matplotlib
plt.ion()
img = plt.imshow(f.T, origin='lower', vmin=-0.3, vmax=0.3)
plt.colorbar()

# Main loop for time evolution
for tn in xrange(1,tmax+1):
	update_src(np.int32(nx), np.int32(ny), np.float32(tn), g_gpu, block=(1,1,1), grid=(1,1))
	update(np.int32(nx), np.int32(ny), c_gpu, f_gpu, g_gpu, block=(256,1,1), grid=(nx*ny/256+1, 1))
	update(np.int32(nx), np.int32(ny), c_gpu, g_gpu, f_gpu, block=(256,1,1), grid=(nx*ny/256+1, 1))

	if tn%10 == 0:
		print("tstep =\t%d/%d (%d %%)" % (tn, tmax, float(tn)/tmax*100))
		cuda.memcpy_dtoh(f, f_gpu)
		img.set_array(f.T)
		plt.draw()
		#savefig('./png/%.5d.png' % tn) 
