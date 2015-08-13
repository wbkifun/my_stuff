from __future__ import division
import numpy as np
import os
from numpy import pi, sqrt, sin, cos
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_


from machine_platform import MachinePlatform
from array_variable import Array, ArrayLike




@raises(Exception)
def test_undefined_machine_type():
    '''
    MachinePlatform : undefined machine_type
    '''
    platform = MachinePlatform('gpu', 'cu', print_on=False)




@raises(Exception)
def test_maismatch_cpu():
    '''
    MachinePlatform : machine_type and code_type are mismatched (CPU, cu)
    '''
    platform = MachinePlatform('CPU', 'cu', print_on=False)




@raises(Exception)
def test_maismatch_gpu():
    '''
    MachinePlatform : machine_type and code_type are mismatched (GPU, f90)
    '''
    platform = MachinePlatform('GPU', 'f90', print_on=False)




def test_cpu_f90():
    '''
    MachinePlatform : CPU, Fortran 90/95
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


    platform = MachinePlatform('CPU', 'f90', print_on=False)
    lib = platform.source_compile(src)
    add = platform.get_function(lib, 'add')

    #----------------------------------------------------------
    # call directly
    #----------------------------------------------------------
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    add(a, b, c)
    a_equal(a+b, c)

    #----------------------------------------------------------
    # call using the array wrapper
    #----------------------------------------------------------
    aa = ArrayLike(platform, a)
    bb = ArrayLike(platform, b)
    cc = Array(platform, aa.shape, aa.dtype)

    add.prepare('IOOO', nx, aa, bb, cc, gsize=nx)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())




def test_cpu_c():
    '''
    MachinePlatform : CPU, C
    '''

    src = '''
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *add_py(PyObject *self, PyObject *args) {
    PyArrayObject *A, *B, *C;
    if (!PyArg_ParseTuple(args, "OOO", &A, &B, &C)) return NULL;

    int nx, i;
    double *a, *b, *c;

    nx = (int)(A->dimensions)[0];
    a = (double*)(A->data);
    b = (double*)(B->data);
    c = (double*)(C->data);

    for (i=0; i<nx; i++) {
        c[i] = a[i] + b[i];
    }

    //Py_INCREF(Py_None);
    //return Py_None;
    Py_RETURN_NONE;
}

static PyMethodDef ufunc_methods[] = {
    {"add", add_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init$MODNAME() {
    Py_InitModule("$MODNAME", ufunc_methods);
    import_array();
}
    '''

    platform = MachinePlatform('CPU', 'c', print_on=False)
    lib = platform.source_compile(src)
    add = platform.get_function(lib, 'add')

    #----------------------------------------------------------
    # call directly
    #----------------------------------------------------------
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    add(a, b, c)
    a_equal(a+b, c)

    #----------------------------------------------------------
    # call using the array wrapper
    #----------------------------------------------------------
    aa = ArrayLike(platform, a)
    bb = ArrayLike(platform, b)
    cc = Array(platform, aa.shape, aa.dtype)

    add.prepare('IOOO', nx, aa, bb, cc, gsize=nx)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())




def test_cpu_cl():
    '''
    MachinePlatform : CPU, OpenCL
    '''

    src = '''
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void add(int nx, __global double *a, __global double *b, __global double *c) {
    int gid = get_global_id(0);

    if (gid >= nx) return;
    c[gid] = a[gid] + b[gid];
}
    '''

    platform = MachinePlatform('CPU', 'cl', print_on=False)
    lib = platform.source_compile(src)
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
    aa = ArrayLike(platform, a)
    bb = ArrayLike(platform, b)
    cc = Array(platform, aa.shape, aa.dtype)

    add.prepare('IOOO', nx, aa, bb, cc, gsize=nx)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())




def test_gpu_cu():
    '''
    MachinePlatform : NVIDIA GPU, CUDA-C
    '''

    src = '''
__global__ void add(int nx, double *a, double *b, double *c) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid >= nx) return;
    c[gid] = a[gid] + b[gid];
}
    '''

    platform = MachinePlatform('NVIDIA GPU', 'cu', print_on=False)
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

    add(nx_cu, a_cu, b_cu, c_cu, block=(256,1,1), grid=(nx//256+1,1))
    cuda.memcpy_dtoh(c, c_cu)
    a_equal(a+b, c)


    #----------------------------------------------------------
    # call using the array wrapper
    #----------------------------------------------------------
    aa = ArrayLike(platform, a)
    bb = ArrayLike(platform, b)
    cc = Array(platform, aa.shape, aa.dtype)

    add.prepare('IOOO', nx, aa, bb, cc, gsize=nx)
    add.prepared_call()
    a_equal(aa.get()+bb.get(), cc.get())