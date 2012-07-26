#!/usr/bin/env python

import pyopencl as cl
import numpy as np
import numpy.linalg as la

kernels = """
__kernel void sum(int n, __global float *a, __global float *b, __global float *c) {
    int gid = get_global_id(0);

    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}
"""

n = 1000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)

prg = cl.Program(ctx, kernels).build()
'''
print 'aaa'
for i in xrange(10000):
	prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)
print 'bbb'
'''
Ls = 256
Gs = n + (Ls - n%Ls)
prg.sum(queue, (Gs,), (Ls,), np.int32(n), a_buf, b_buf, dest_buf)
a_plus_b = np.empty_like(a)
cl.enqueue_read_buffer(queue, dest_buf, a_plus_b)

print la.norm(a_plus_b - (a+b))
assert la.norm(a_plus_b - (a+b)) == 0
