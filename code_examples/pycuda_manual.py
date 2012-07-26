import pycuda.driver as cuda
import numpy as np

cuda.init()
dev = cuda.Device(1)
ctx = dev.make_context()

kernels = '''
__global__ void bla(int nx, float *a, float *b) {
    int tid = threadIdx.x;
    if (tid < nx) {
        b[tid] = a[tid];
    }
}
'''

from pycuda.compiler import SourceModule
mod = SourceModule(kernels)
bla = mod.get_function('bla')

nx = 256
a = np.ones(nx, 'f')
b = np.zeros(nx, 'f')

a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
bla(np.int32(nx), a_gpu, b_gpu, block=(256,1,1), grid=(1,1))
cuda.memcpy_dtoh(b, b_gpu)
print np.linalg.norm(a - b) == 0

ctx.pop()
