#!/usr/bin/env python
#---------------------------------------------------------------------------
# File Name : vecadd_pyopencl_multi_gpu.py
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
# Step 2: PyOpenCL
# Step 3: Multi-GPU with PyOpenCL
#---------------------------------------------------------------------------

import numpy as np
import pyopencl as cl


# Host(CPU) result
nx = 2400
a = np.random.rand(nx).astype(np.float32)
b = np.random.rand(nx).astype(np.float32)
c = a[:] + b[:]


# GPU result
platforms = cl.get_platforms()
devices = platforms[0].get_devices(cl.device_type.GPU)	# assume single platform
ctx = cl.Context(devices)
queues = [cl.CommandQueue(ctx, device) for device in devices]
prg = cl.Program(ctx, open('./vecadd.cl').read()).build()

ngpu = len(devices)
snx = nx / len(devices)

mflags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
a_gpus, b_gpus, c_gpus = [], [], []
for i, queue in enumerate(queues):
	a_gpus.append(cl.Buffer(ctx, mflags, hostbuf=a[snx*i:snx*(i+1)]))
	b_gpus.append(cl.Buffer(ctx, mflags, hostbuf=b[snx*i:snx*(i+1)]))
	c_gpus.append(cl.Buffer(ctx, cl.mem_flags.READ_WRITE, c.nbytes/ngpu))
c_from_gpu = np.zeros_like(c)

for queue, a_gpu, b_gpu, c_gpu in zip(queues, a_gpus, b_gpus, c_gpus):
	prg.vecadd(queue, (256*4,), (256,), np.int32(snx), a_gpu, b_gpu, c_gpu)

for i, queue in enumerate(queues):
	cl.enqueue_read_buffer(queue, c_gpus[i], c_from_gpu[snx*i:snx*(i+1)])


# Verify
assert np.linalg.norm(c - c_from_gpu) == 0
