#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


static PyObject *daxpy_py(PyObject *self, PyObject *args) {
	double a;
	PyArrayObject *X, *Y;
	if (!PyArg_ParseTuple(args, "dOO", &a, &X, &Y)) return NULL;

	int n, i;
	double *x, *y;

	n = (int)(X->dimensions)[0];
	x = (double*)(X->data);
	y = (double*)(Y->data);

	for (i = 0; i<n; i++) {
		y[i] = a*x[i] + y[i];
	}

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
