#------------------------------------------------------------------------------
# filename  : test_daxpy.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.8.22    start
#             2015.9.23    modify the case of CPU-C with f2py
#             2015.11.4    MachinePlatform class splits into Device classes
#
#
# description:
#   Test the MachinePlatform class with DAXPY(Double-precision A*X + B)
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import os
from numpy import pi, sqrt, sin, cos
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_


from array_variable import Array, ArrayAs




def run_and_check(platform, src):
    n = 2**20
    a = np.random.rand()
    x = np.random.rand(n)
    y = np.random.rand(n)
    ref = a*x + y

    xx = ArrayAs(platform, x)
    yy = ArrayAs(platform, y)

    lib = platform.source_compile(src)
    func = platform.get_function(lib, 'daxpy')

    # (int32, float64, float64 array, float64 array)
    func.prepare('idoo', n, a, xx, yy, gsize=n)
    func.prepared_call()

    #a_equal(ref, yy.get())
    aa_equal(ref, yy.get(), 15)




def test_cpu_f90():
    '''
    DAXPY: CPU, Fortran 90/95
    '''

    src = '''
SUBROUTINE daxpy(n, a, x, y)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a
  REAL(8), DIMENSION(n), INTENT(IN) :: x
  REAL(8), DIMENSION(n), INTENT(INOUT) :: y

  INTEGER :: i

  DO i=1,n
    y(i) = a*x(i) + y(i)
  END DO
END SUBROUTINE
    '''

    from device_platform import CPU_F90
    platform = CPU_F90()
    run_and_check(platform, src)




def test_cpu_c():
    '''
    DAXPY: CPU, C
    '''

    src = '''
#include <math.h>

void daxpy(int n, double a, double *x, double *y) {
    // size and intent of array arguments for f2py
    // x :: n, in
    // y :: n, inout

    int i;

    for (i=0; i<n; i++) {
        y[i] = a*x[i] + y[i];
    }
}
    '''

    from device_platform import CPU_C
    platform = CPU_C()
    run_and_check(platform, src)




def test_cpu_opencl():
    '''
    DAXPY: CPU, OpenCL
    '''

    src = '''
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void daxpy(int n, double a, __global double *x, __global double *y) {
    int gid = get_global_id(0);

    if (gid >= n) return;
    y[gid] = a*x[gid] + y[gid];
}
    '''

    # Prevent a warning message when a Program.build() is called
    os.environ['PYOPENCL_NO_CACHE'] = '1'

    from device_platform import CPU_OpenCL
    platform = CPU_OpenCL()
    run_and_check(platform, src)




def test_nvidia_gpu_cuda():
    '''
    DAXPY: NVIDIA GPU, CUDA-C
    '''

    src = '''
__global__ void daxpy(int n, double a, double *x, double *y) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid >= n) return;
    y[gid] = a*x[gid] + y[gid];
}
    '''

    from device_platform import NVIDIA_GPU_CUDA
    platform = NVIDIA_GPU_CUDA(0)
    run_and_check(platform, src)
