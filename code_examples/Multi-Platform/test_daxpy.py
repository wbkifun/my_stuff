#------------------------------------------------------------------------------
# filename  : test_daxpy.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.8.22    start
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

    lib = platform.source_compile(src)
    func = platform.get_function(lib, 'daxpy')
    func.prepare('IdOO', n, a, xx, yy, gsize=n)  # (optional int32, float64, float64 array, float64 array, global thread size)
    func.prepared_call()

    #a_equal(ref, yy.get())
    aa_equal(ref, yy.get(), 15)




def test_cpu_f90():
    '''
    DAXPY : CPU, Fortran 90/95
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

    platform = MachinePlatform('CPU', 'f90', print_on=False)
    run_and_check(platform, src)




def test_cpu_c():
    '''
    DAXPY : CPU, C
    '''

    src = '''
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *daxpy_py(PyObject *self, PyObject *args) {
    double a;
    PyArrayObject *X, *Y;
    if (!PyArg_ParseTuple(args, "dOO", &a, &X, &Y)) return NULL;

    int n, i;
    double *x, *y;

    n = (int)(X->dimensions)[0];
    x = (double*)(X->data);
    y = (double*)(Y->data);

    for (i=0; i<n; i++) {
        y[i] = a*x[i] + y[i];
    }

    Py_RETURN_NONE;
}

static PyMethodDef ufunc_methods[] = {
    {"daxpy", daxpy_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init$MODNAME() {
    Py_InitModule("$MODNAME", ufunc_methods);
    import_array();
}
    '''

    platform = MachinePlatform('CPU', 'c', print_on=False)
    run_and_check(platform, src)




def test_cpu_cl():
    '''
    DAXPY : CPU, OpenCL
    '''

    src = '''
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void daxpy(int n, double a, __global double *x, __global double *y) {
    int gid = get_global_id(0);

    if (gid >= n) return;
    y[gid] = a*x[gid] + y[gid];
}
    '''

    platform = MachinePlatform('CPU', 'cl', print_on=False)
    run_and_check(platform, src)




def test_nvidia_gpu_cu():
    '''
    DAXPY : NVIDIA GPU, CUDA-C
    '''

    src = '''
__global__ void daxpy(int n, double a, double *x, double *y) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid >= n) return;
    y[gid] = a*x[gid] + y[gid];
}
    '''

    platform = MachinePlatform('NVIDIA GPU', 'cu', print_on=False)
    run_and_check(platform, src)
