import numpy as np
import pyopencl as cl
 

def vecadd(a, b):
    return a + b
 

kernels = """
__kernel void vecadd(int n, __global float *a, __global float *b, __global float *c) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}
"""

n = 5
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = vecadd(a, b)
 
#ctx = cl.create_some_context()
#queue = cl.CommandQueue(ctx)
platforms = cl.get_platforms()
devices = platforms[0].get_devices()
ctx = cl.Context(devices)
queue = cl.CommandQueue(ctx, devices[0])
prg = cl.Program(ctx, kernels).build()

mf = cl.mem_flags
a_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_dev = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
c_from_dev = np.zeros_like(c)

ls = 256                # local work size
gs = n + (ls - n%ls)    # global work size
prg.vecadd(queue, (gs,), (ls,), np.int32(n), a_dev, b_dev, c_dev)
cl.enqueue_read_buffer(queue, c_dev, c_from_dev)
 
print 'OK!'if np.linalg.norm(c - c_from_dev) == 0 else 'Failed!'
