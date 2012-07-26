#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *fib(PyObject *self, PyObject *args) {
	PyArrayObject *A;
	if (!PyArg_ParseTuple(args, "O", &A)) return NULL;

	int n, i;
	double *a;
	n = (int)(A->dimensions)[0];
	a = (double*)(A->data);

	for( i=1; i<n+1; i++ ) {
		if( i==0 ) a[i] = 0;
		if( i==1 ) a[i] = 1;
		else a[i] = a[i-1] + a[i-2];
	}

   	Py_INCREF(Py_None);
   	return Py_None;
}

static PyMethodDef cfunc_methods[] = {
	{"fib", fib, METH_VARARGS, ""},
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC initfib1() {
	Py_InitModule3("fib1", cfunc_methods, "");
	import_array();
}
