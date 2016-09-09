#------------------------------------------------------------------------------
# filename  : test_device_platform.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.10.29   start
#             2016.3.10    insert Array and ArrayAs as members of the class
#             2016.8.29    modify source_module.py: 
#                          get_module() -> compile_using_f2py()
#------------------------------------------------------------------------------

import numpy as np
import os
from numpy import pi, sqrt, sin, cos
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_
import io, sys




def capture(func):
    def wrapper(*args, **kwargs):
        capturer1 = io.StringIO()
        capturer2 = io.StringIO()
        old_stdout, sys.stdout = sys.stdout, capturer1
        old_stderr, sys.stderr = sys.stderr, capturer2

        ret = func(*args, **kwargs)

        sys.stdout, sys.stderr = old_stdout, old_stderr
        out = capturer1.getvalue().rstrip('\n')
        err = capturer2.getvalue().rstrip('\n')

        return ret, out, err

    return wrapper




def test_cpu_f90():
    '''
    CPU_F90: c = a + b
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
    from device_platform import DevicePlatform

    platform = DevicePlatform('CPU', 'F90')
    lib, out, err = capture(platform.source_compile)(src, compiler='gnu', flags='', opt_flags='-O3')
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
    aa = platform.ArrayAs(a)
    bb = platform.ArrayAs(b)
    cc = platform.Array(aa.size, aa.dtype)

    add.prepare('iooo', nx, aa, bb, cc)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())




def test_cpu_c():
    '''
    CPU_C: c = a + b
    '''

    src = '''
void add(int nx, double *a, double *b, double *c) {
    // size and intent of array arguments for f2py
    // a :: nx, in
    // b :: nx, in
    // c :: nx, inout

    int i;

    for (i=0; i<nx; i++) {
        c[i] = a[i] + b[i];
    }
}
    '''
    from device_platform import CPU_C

    platform = CPU_C()
    lib, out, err = capture(platform.source_compile)(src, compiler='gnu', flags='', opt_flags='-O3')
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
    aa = platform.ArrayAs(a)
    bb = platform.ArrayAs(b)
    cc = platform.Array(aa.size, aa.dtype)

    add.prepare('iooo', nx, aa, bb, cc)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())




def test_nvidia_gpu_cuda():
    '''
    NVIDIA_GPU_CUDA: c = a + b
    '''

    src = '''
__global__ void add(int shift_gid, int nx, double *a, double *b, double *c) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x + shift_gid;

    if (gid >= nx) return;
    c[gid] = a[gid] + b[gid];
}
    '''
    from device_platform import NVIDIA_GPU_CUDA

    platform = NVIDIA_GPU_CUDA(device_number=0)
    lib = platform.source_compile(src)
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

    add(np.int32(0), nx_cu, a_cu, b_cu, c_cu, block=(512,1,1), grid=(nx//512+1,1))
    cuda.memcpy_dtoh(c, c_cu)
    a_equal(a+b, c)

    #----------------------------------------------------------
    # call using the array wrapper
    #----------------------------------------------------------
    aa = platform.ArrayAs(a)
    bb = platform.ArrayAs(b)
    cc = platform.Array(aa.size, aa.dtype)

    add.prepare('iooo', nx, aa, bb, cc, gsize=nx)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())




def test_cpu_opencl():
    '''
    CPU_OpenCL: c = a + b
    '''

    src = '''
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void add(int nx, __global double *a, __global double *b, __global double *c) {
    int gid = get_global_id(0);

    if (gid >= nx) return;
    c[gid] = a[gid] + b[gid];
}
    '''
    import os
    from device_platform import CPU_OPENCL

    platform = CPU_OPENCL()
    lib, out, err = capture(platform.source_compile)(src)
    add = platform.get_function(lib, 'add')

    #-------------------------------------------
    # call directly
    #-------------------------------------------
    nx = 1000000
    #nx = 500000000
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

    #add(queue, (nx,), None, nx_cl, a_cl, b_cl, c_cl)
    add(queue, (nx,), (1,), nx_cl, a_cl, b_cl, c_cl)
    #add(queue, (nx//2,), (2,), nx_cl, a_cl, b_cl, c_cl)
    #cl.enqueue_read_buffer(queue, c_cl, c)     # deprecated
    cl.enqueue_copy(queue, c, c_cl)
    a_equal(a+b, c)


    #----------------------------------------------------------
    # call using the array wrapper
    #----------------------------------------------------------
    aa = platform.ArrayAs(a)
    bb = platform.ArrayAs(b)
    cc = platform.Array(aa.size, aa.dtype)

    add.prepare('iooo', nx, aa, bb, cc, gsize=nx)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())
