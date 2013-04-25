#!/usr/bin/env python

import numpy as np
import pyopencl as cl


# Host(CPU) result
nx = 1024 * 50
a = np.random.rand(nx).astype(np.float32)
b = np.random.rand(nx).astype(np.float32)
d = np.arange(nx, dtype=np.float32)
c = a[:] + b[:] + d[:]


# GPU result
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
kernels = """
//__constant float d[1024];

__kernel void vecadd(int nx, __global const float *a, __global const float *b, __global float *c, __constant float *d) {
	int gid = get_global_id(0);
	
	if( gid < nx ) {
		c[gid] = a[gid] + b[gid] + d[gid];
	}
}
"""
prg = cl.Program(ctx, kernels).build()

mflags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
a_gpu = cl.Buffer(ctx, mflags, hostbuf=a)
b_gpu = cl.Buffer(ctx, mflags, hostbuf=b)
c_gpu = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, c.nbytes)
c_from_gpu = np.zeros_like(c)
d_gpu = cl.Buffer(ctx, mflags, hostbuf=d)
#d_gpu = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, d.nbytes)
#cl.enqueue_write_buffer(queue, d_gpu, d)

prg.vecadd(queue, (nx,), (256,), np.int32(nx), a_gpu, b_gpu, c_gpu, d_gpu)
cl.enqueue_read_buffer(queue, c_gpu, c_from_gpu)


# Verify
assert np.linalg.norm(c - c_from_gpu) == 0
