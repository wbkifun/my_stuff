#!/usr/bin/env python
#---------------------------------------------------------------------------
# File Name : vecadd_pyopencl_ext_file.py
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
#---------------------------------------------------------------------------

import numpy as np
import pyopencl as cl


# Host(CPU) result
nx = 2000
a = np.random.rand(nx).astype(np.float32)
b = np.random.rand(nx).astype(np.float32)
c = a[:] + b[:]


# GPU result
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, open('./vecadd.cl').read()).build()

mflags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
a_gpu = cl.Buffer(ctx, mflags, hostbuf=a)
b_gpu = cl.Buffer(ctx, mflags, hostbuf=b)
c_gpu = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, c.nbytes)
c_from_gpu = np.zeros_like(c)

prg.vecadd(queue, (256*4,), (256,), np.int32(nx), a_gpu, b_gpu, c_gpu)
cl.enqueue_read_buffer(queue, c_gpu, c_from_gpu)


# Verify
assert np.linalg.norm(c - c_from_gpu) == 0
