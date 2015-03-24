from __future__ import division
import numpy as np
import os
from numpy import pi, sqrt, sin, cos
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_



from source_module import get_module_f90, get_module_c



def test_get_module_f90():
    # setup
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)


    src_f = '''
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

    mod_f = get_module_f90(src_f)
    add_f = mod_f.add

    add_f(a, b, c)
    a_equal(a+b, c)




def test_get_module_c():
    # setup
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    src_c = '''
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *add(PyObject *self, PyObject *args) {
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
    {"add", add, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init$MODNAME() {
    Py_InitModule("$MODNAME", ufunc_methods);
    import_array();
}
    '''

    mod_c = get_module_c(src_c)
    add_c = mod_c.add

    add_c(a, b, c)
    a_equal(a+b, c)
