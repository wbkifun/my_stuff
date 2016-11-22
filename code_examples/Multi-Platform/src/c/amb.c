#include <Python.h>
#include <numpy/arrayobject.h>
#include "param1.h"
#include "amb_ext1.h"

void amb(int nx, double *a, double *b, double *c) {
	// size and intent of array arguments for f2py
	// a :: nx, in
	// b :: nx, in
	// c :: nx, inout
	
	int i;

	bmc(nx, LLL, b, c);

	for (i=0; i<nx; i++) {
		c[i] = KK*a[i] + c[i];
	}
}

static PyObject *amb(PyObject *self, PyObject *args) {
	int nx;
	PyArrayObject *A, *B, *C;
	if (!PyArg_ParseTuple(args, "iOO", &nx, &A, &B, &C)) return NULL;

	int i;
	double *a, *b, *c;

	a = (double*)(A->data);
	b = (double*)(B->data);
	c = (double*)(C->data);

	//-------------------------------------------------------------------------
	// Body
	//-------------------------------------------------------------------------
	bmc(nx, LLL, b, c);

	for (i=0; i<nx; i++) {
		c[i] = KK*a[i] + c[i];
	}
	//-------------------------------------------------------------------------

	Py_DECREF(A);
	Py_DECREF(B);
	Py_DECREF(C);
	Py_RETURN_NONE;
}


static PyMethodDef ufunc_methods[] = {
	    {"daxpy", daxpy_py, METH_VARARGS, ""},
		    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initdaxpy() {
	    Py_InitModule("daxpy", ufunc_methods);
		    import_array();
}
