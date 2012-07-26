import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
 

def vecadd(a, b):
    return a + b
 

kernels = """
__global__ void vecadd(int n, float *a, float *b, float *c) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}
"""

n = 5
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = vecadd(a, b)
 
from pycuda.compiler import SourceModule
mod = SourceModule(kernels)
vecadd = mod.get_function('vecadd')

a_dev = cuda.to_device(a)
b_dev = cuda.to_device(b)
c_dev = cuda.mem_alloc(c.nbytes)
c_from_dev = np.zeros_like(c)

ls = 256                # local work size
gs = n + (ls - n%ls)    # global work size
vecadd(np.int32(n), a_dev, b_dev, c_dev, block=(ls,1,1), grid=(gs/ls,1))
cuda.memcpy_dtoh(c_from_dev, c_dev)
 
if np.linalg.norm(c - c_from_dev) == 0:
    print 'OK!'
else:
    print 'Failed!'