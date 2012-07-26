#!/usr/bin/env python
#---------------------------------------------------------------------------
# File Name : wave2d_pyopencl.py
#
# Author : Ki-Hwan Kim (wbkifun@nate.com)
# 
# Written date :	2011. 6. 23
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
# Step 3: Utilize the GPU using PyOpenCL
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

kernels ="""
__kernel void update(int nx, int ny, __global float *c, __global float *f, __global float *g) {
	int gid = get_global_id(0);
	int i = gid/ny;
	int j = gid%ny;
	
	if( i>0 && i<nx-1 && j>0 && j<ny-1 )
		f[gid] = c[gid]*(g[gid+ny] + g[gid-ny] + g[gid+1] + g[gid-1] - 4*g[gid]) + 2*g[gid] - f[gid];
}


__kernel void update_src(int nx, int ny, float tn, __global float *f) {
	f[300*ny+400] += sin(0.1*tn);
}
"""
prg = cl.Program(ctx, kernels).build()


nx, ny = 1000, 800
tmax = 500
f = np.zeros((nx,ny), dtype=np.float32)
c = np.ones_like(f) * 0.25
mflags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
f_gpu = cl.Buffer(ctx, mflags, hostbuf=f)
g_gpu = cl.Buffer(ctx, mflags, hostbuf=f)
c_gpu = cl.Buffer(ctx, mflags, hostbuf=c)

# To plot using matplotlib
plt.ion()
img = plt.imshow(f.T, origin='lower', vmin=-0.3, vmax=0.3)
plt.colorbar()

# Main loop for time evolution
for tn in xrange(1,tmax+1):
	prg.update_src(queue, (1,), (1,), np.int32(nx), np.int32(ny), np.float32(tn), g_gpu)
	prg.update(queue, (nx*ny,), (256,), np.int32(nx), np.int32(ny), c_gpu, f_gpu, g_gpu)
	prg.update(queue, (nx*ny,), (256,), np.int32(nx), np.int32(ny), c_gpu, g_gpu, f_gpu)

	if tn%10 == 0:
		print("tstep =\t%d/%d (%d %%)" % (tn, tmax, float(tn)/tmax*100))
		cl.enqueue_read_buffer(queue, f_gpu, f)
		img.set_array(f.T)
		plt.draw()
		#savefig('./png/%.5d.png' % tn) 
