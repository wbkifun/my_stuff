#b------------------------------------------------------------------------------
# filename  : daxpy_pycuda.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.8.18     start
#------------------------------------------------------------------------------

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal


# kernel compile and import
kernel = '''
__global__ void daxpy(int nx, double a, double *x, double *y) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < nx) y[idx] = a*x[idx] + y[idx];
}
'''
from pycuda.compiler import SourceModule, compile
#mod = SourceModule(kernel, cache_dir='./')

cubin = compile(kernel, cache_dir='./')
#mod = cuda.module_from_buffer(cubin)

with open('daxpy.cubin', 'wb') as f: f.write(cubin)
mod = cuda.module_from_file('daxpy.cubin')
daxpy = mod.get_function('daxpy')

dev = pycuda.autoinit.device
print(dev.compute_capability())
#cuda.device_attribute['COMPUTE_CAPABILITY_MAJOR']
#cuda.device_attribute['COMPUTE_CAPABILITY_MINOR']


# setup
nx = 2**20

# allocation
a = np.random.rand()
x = np.random.rand(nx)
y = np.random.rand(nx)
y1 = np.zeros_like(y)
y2 = np.zeros_like(y)

# allocation on GPU
x_gpu = cuda.to_device(x)
y_gpu = cuda.to_device(y)

# run
daxpy(np.int32(nx), np.float64(a), x_gpu, y_gpu, block=(512,1,1), grid=(nx//512+1,1))

# check
y1[:] = a*x + y
cuda.memcpy_dtoh(y2, y_gpu)
#a_equal(y1, y2)
aa_equal(y1, y2, 15)
