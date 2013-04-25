#!/usr/bin/env python

import pyopencl as cl
import numpy as np

kernels = """
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void func(__global double *a, __global double *b, __global double *c) {
	int gid = get_global_id(0);
    a[gid] += 0.4 * b[gid];
    //a[gid] = fma(0.4, b[gid], a[gid]);
}
"""

nx = 100
a = np.random.rand(nx).astype(np.float64)
b = np.random.rand(nx).astype(np.float64)
c = np.random.rand(nx).astype(np.float64)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

prg = cl.Program(ctx, kernels).build(options=['-cl-opt-disable'])
prg.func(queue, (nx,), (nx,), a_buf, b_buf, c_buf)

a_from_gpu = np.empty_like(a)
cl.enqueue_copy(queue, a_from_gpu, a_buf)

a[:] += 0.4 * b[:]
print np.linalg.norm(a - a_from_gpu)
