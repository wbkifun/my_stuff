from __future__ import division
import numpy as np
import os
from numpy import pi, sqrt, sin, cos
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

import atexit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda.init()
dev = cuda.Device(0)
ctx = dev.make_context()
atexit.register(ctx.pop)


src_cu = '''
#define NX 3
#define NY 4

__global__ void compare_order(double* arr_2d, double* arr_1d) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= NX*NY) return;

    arr_1d[tid] = arr_2d[tid];
}
'''
mod_cu = SourceModule(src_cu)



def test_compare_order():
    '''
    compare_order between C(row-major), F(column-major)
    '''
    compare_order = mod_cu.get_function('compare_order')


    nx, ny = 3, 4
    f_1d = np.arange(nx*ny, dtype='f8')
    f_2d_C = f_1d.reshape((nx,ny), order='C')
    f_2d_F = f_1d.reshape((nx,ny), order='F')

    print ''
    print 'f_1d_C\n\n', f_1d
    print 'f_2d_C\n', f_2d_C
    print 'f_2d_F\n', f_2d_F

    print ''
    print 'after cuda'
    ret_f_1d = np.zeros_like(f_1d)
    f_1d_gpu = cuda.mem_alloc_like(f_1d)

    f_2d_C_gpu = cuda.to_device(f_2d_C)
    compare_order(f_2d_C_gpu, f_1d_gpu, block=(nx*ny,1,1), grid=(1,1))
    cuda.memcpy_dtoh(ret_f_1d, f_1d_gpu)
    print 'f_1d from f_2d_C\n', ret_f_1d

    f_2d_F_gpu = cuda.to_device(f_2d_F)
    compare_order(f_2d_F_gpu, f_1d_gpu, block=(nx*ny,1,1), grid=(1,1))
    cuda.memcpy_dtoh(ret_f_1d, f_1d_gpu)
    print 'f_1d from f_2d_F\n', ret_f_1d
