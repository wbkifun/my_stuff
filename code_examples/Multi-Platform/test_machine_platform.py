#------------------------------------------------------------------------------
# filename  : test_machine_platform.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.9.23    revise
#             2015.11.2    heterogeneous machines
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import os
from numpy import pi, sqrt, sin, cos
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_


from machine_platform import MachinePlatform
from array_variable import Array, ArrayAs




def run_and_check(platform, src):
    n = 2**20
    a = np.random.rand()
    x = np.random.rand(n)
    y = np.random.rand(n)
    ref = a*x + y

    xx = ArrayAs(platform, x)
    yy = ArrayAs(platform, y)

    src_list = [globals()['src_%s'%suffix] for suffix in platform.code_types]
    lib_list = platform.source_compile(src_list)
    func_list = platform.get_function(lib_list, 'daxpy')
    platform.func_prepare(func_list, 'idoo', n, a, xx, yy)  # (int32, float64, float64 array, float64 array)
    platform.func_prepared_call(func_list)

    #a_equal(ref, yy.get())
    aa_equal(ref, yy.get(), 15)





@raises(Exception)
def test_undefined_device_type():
    '''
    MachinePlatform : undefined device_type
    '''
    platform = MachinePlatform([('KPU','f90','all')])




@raises(Exception)
def test_mismatch_cpu():
    '''
    MachinePlatform : machine_type and code_type are mismatched (CPU, cu)
    '''
    platform = MachinePlatform([('CPU','cu','all')])




@raises(Exception)
def test_mismatch_gpu():
    '''
    MachinePlatform : machine_type and code_type are mismatched (NVIDIA_GPU, f90)
    '''
    try:
        platform = MachinePlatform([('NVIDIA_GPU','f90','all')])
    except Exception,e:
        print e




def test_gpu_cpu():
    '''
    MachinePlatform: Heterogeneous (NVIDIA_GPU_CU, CPU_F90)
    '''

    src = '''
SUBROUTINE add(nx, a, b, c)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx
  REAL(8), DIMENSION(nx), INTENT(IN) :: a, b
  REAL(8), DIMENSION(nx), INTENT(INOUT) :: c

  INTEGER :: ii

  DO ii=1,nx
    c(ii) = a(ii) + b(ii)
  END DO
END SUBROUTINE
    '''

    pyf = '''
python module $MODNAME
  interface
    subroutine add(n,a,b,c)
      integer, required, intent(in) :: n
      double precision, intent(in) :: a(n), b(n)
      double precision, intent(inout) :: c(n)
    end subroutine
  end interface
end python module
    '''

    platform = MachinePlatform('CPU', 'f90', print_on=False)
    lib = platform.source_compile(src, pyf)
    add = platform.get_function(lib, 'add')

    #----------------------------------------------------------
    # call directly
    #----------------------------------------------------------
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    add(nx, a, b, c)
    a_equal(a+b, c)

    #----------------------------------------------------------
    # call using the array wrapper
    #----------------------------------------------------------
    aa = ArrayAs(platform, a)
    bb = ArrayAs(platform, b)
    cc = Array(platform, aa.size, aa.dtype)

    add.prepare('iooo', nx, aa, bb, cc)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())




def test_cpu_c():
    '''
    MachinePlatform: CPU, C
    '''

    src = '''
void add(int nx, double *a, double *b, double *c) {
    int i;

    for (i=0; i<nx; i++) {
        c[i] = a[i] + b[i];
    }
}
    '''

    pyf = '''
python module $MODNAME
  interface
    subroutine add(n,a,b,c)
      intent(c) :: add
      intent(c)    ! Adds to all following definitions
      integer, required, intent(in) :: n
      double precision, intent(in) :: a(n), b(n)
      double precision, intent(inout) :: c(n)
    end subroutine
  end interface
end python module
    '''

    platform = MachinePlatform('CPU', 'c', print_on=False)
    lib = platform.source_compile(src, pyf)
    add = platform.get_function(lib, 'add')

    #----------------------------------------------------------
    # call directly
    #----------------------------------------------------------
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    add(nx, a, b, c)
    a_equal(a+b, c)

    #----------------------------------------------------------
    # call using the array wrapper
    #----------------------------------------------------------
    aa = ArrayAs(platform, a)
    bb = ArrayAs(platform, b)
    cc = Array(platform, aa.size, aa.dtype)

    add.prepare('iooo', nx, aa, bb, cc)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())




def test_cpu_cl():
    '''
    MachinePlatform: CPU, OpenCL
    '''

    src = '''
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void add(int nx, __global double *a, __global double *b, __global double *c) {
    int gid = get_global_id(0);

    if (gid >= nx) return;
    c[gid] = a[gid] + b[gid];
}
    '''

    platform = MachinePlatform('CPU', 'cl', print_on=False)
    lib = platform.source_compile(src, None)
    add = platform.get_function(lib, 'add')

    #-------------------------------------------
    # call directly
    #-------------------------------------------
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    cl = platform.cl
    ctx = platform.context
    queue = platform.queue
    mf = cl.mem_flags
    read_copy = mf.READ_ONLY | mf.COPY_HOST_PTR

    nx_cl = np.int32(nx)
    a_cl = cl.Buffer(ctx, read_copy, hostbuf=a)
    b_cl = cl.Buffer(ctx, read_copy, hostbuf=b)
    c_cl = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

    add(queue, (nx,), None, nx_cl, a_cl, b_cl, c_cl)
    #cl.enqueue_read_buffer(queue, c_cl, c)     # deprecated
    cl.enqueue_copy(queue, c, c_cl)
    a_equal(a+b, c)


    #----------------------------------------------------------
    # call using the array wrapper
    #----------------------------------------------------------
    aa = ArrayAs(platform, a)
    bb = ArrayAs(platform, b)
    cc = Array(platform, aa.size, aa.dtype)

    add.prepare('iooo', nx, aa, bb, cc)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())




def test_nvidia_gpu_cu():
    '''
    MachinePlatform: NVIDIA GPU, CUDA-C
    '''

    src = '''
__global__ void add(int nx, double *a, double *b, double *c) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid >= nx) return;
    c[gid] = a[gid] + b[gid];
}
    '''

    platform = MachinePlatform('NVIDIA GPU', 'cu', print_on=False)
    lib = platform.source_compile(src, None)
    add = platform.get_function(lib, 'add')

    #-------------------------------------------
    # call directly
    #-------------------------------------------
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    cuda = platform.cuda

    nx_cu = np.int32(nx)
    a_cu = cuda.to_device(a)
    b_cu = cuda.to_device(b)
    c_cu = cuda.mem_alloc_like(c)

    add(nx_cu, a_cu, b_cu, c_cu, block=(256,1,1), grid=(nx//256+1,1))
    cuda.memcpy_dtoh(c, c_cu)
    a_equal(a+b, c)


    #----------------------------------------------------------
    # call using the array wrapper
    #----------------------------------------------------------
    aa = ArrayAs(platform, a)
    bb = ArrayAs(platform, b)
    cc = Array(platform, aa.size, aa.dtype)

    add.prepare('iooo', nx, aa, bb, cc)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())
