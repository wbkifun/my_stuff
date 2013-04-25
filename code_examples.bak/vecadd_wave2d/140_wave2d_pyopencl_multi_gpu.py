#!/usr/bin/env python
#---------------------------------------------------------------------------
# File Name : wave2d_pyopencl_multi_gpu.py
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
# Step 4: Utilize the Multi-GPU using PyOpenCL
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl


def exchange_boundary(snx, ny, queues, f_gpus, tmp_hs, tmp_ts):
	ngpu = len(queues)

	for i, queue in enumerate(queues):
		if i>0: 
			cl.enqueue_read_buffer(queue, f_gpus[i], tmp_hs[i], device_offset=ny*4)
		if i<ngpu-1: 
			cl.enqueue_read_buffer(queue, f_gpus[i], tmp_ts[i], device_offset=(snx-2)*ny*4)

	for i, queue in enumerate(queues):
		if i>0: 
			cl.enqueue_write_buffer(queue, f_gpus[i], tmp_ts[i-1])
		if i<ngpu-1: 
			cl.enqueue_write_buffer(queue, f_gpus[i], tmp_hs[i+1], device_offset=(snx-1)*ny*4)


# Setup
nx, ny = 1200, 800	# nx must be mutiple of ngpu
tmax = 500
f = np.zeros((nx,ny), dtype=np.float32)
c = np.ones_like(f) * 0.25


# Platform, Device, Context, Queue, Program
platforms = cl.get_platforms()
devices = platforms[0].get_devices(cl.device_type.GPU)	# assume single platform
ctx = cl.Context(devices)
queues = [cl.CommandQueue(ctx, device) for device in devices]
prg = cl.Program(ctx, open('./wave2d_core.cl').read()).build()

ngpu = len(devices)
snx = nx / len(devices)


# Allocate in the GPU device memory
mflags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
f_gpus, g_gpus, c_gpus = [], [], []
for i, queue in enumerate(queues):
	f_gpus.append(cl.Buffer(ctx, mflags, hostbuf=f[snx*i:snx*(i+1),:]))
	g_gpus.append(cl.Buffer(ctx, mflags, hostbuf=f[snx*i:snx*(i+1),:]))
	c_gpus.append(cl.Buffer(ctx, mflags, hostbuf=c[snx*i:snx*(i+1),:]))


# To exchange the boundary values
tmp_hs = [np.zeros(ny, dtype=np.float32) for queue in queues]	# head
tmp_ts = [np.zeros(ny, dtype=np.float32) for queue in queues]	# tail


# To plot using matplotlib
plt.ion()
img = plt.imshow(f.T, origin='lower', vmin=-0.3, vmax=0.3)
plt.colorbar()


# Main loop for time evolution
for tn in xrange(1,tmax+1):
	# update source in GPU 0
	prg.update_src(queues[0], (1,), (1,), np.int32(snx), np.int32(ny), np.float32(tn), g_gpus[0])

	# update f 
	for queue, f_gpu, g_gpu, c_gpu in zip(queues, f_gpus, g_gpus, c_gpus):
		prg.update(queue, (snx*ny,), (256,), np.int32(snx), np.int32(ny), c_gpu, f_gpu, g_gpu)
	exchange_boundary(snx, ny, queues, f_gpus, tmp_hs, tmp_ts)

	# update g
	for queue, f_gpu, g_gpu, c_gpu in zip(queues, f_gpus, g_gpus, c_gpus):
		prg.update(queue, (snx*ny,), (256,), np.int32(snx), np.int32(ny), c_gpu, g_gpu, f_gpu)
	exchange_boundary(snx, ny, queues, g_gpus, tmp_hs, tmp_ts)


	if tn%10 == 0:
		print("tstep =\t%d/%d (%d %%)" % (tn, tmax, float(tn)/tmax*100))
		for i, queue, f_gpu in zip(range(ngpu), queues, f_gpus):
			cl.enqueue_read_buffer(queue, f_gpu, f[snx*i:snx*(i+1),:])
		img.set_array(f.T)
		plt.draw()
		#savefig('./png/%.5d.png' % tn) 
