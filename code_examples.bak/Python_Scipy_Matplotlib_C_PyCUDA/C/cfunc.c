#include <Python.h>
#include <numpy/arrayobject.h>


static PyObject *diff(PyObject *self, PyObject *args) {
	int Nx, Ny;
	float dx;
	PyArrayObject *A, *dA;

	if (!PyArg_ParseTuple(args, "iifOO", &Nx, &Ny, &dx, &A, &dA )) return NULL;


	int i, j, idx;
	float *a, *da;
	a = (float*)(A->data);
	da = (float*)(dA->data);

	for ( i=0; i<Nx-1; i++ ) {
		for ( j=0; j<Ny-1; j++ ) {
			idx = i*Ny + j;
			da[idx] = (1/dx)*( a[idx+Ny+1] - a[idx] );
		}
	}

   	Py_INCREF(Py_None);
   	return Py_None;
}

/* =============================================================
 * method table listing
 * module's initialization
============================================================= */
static char diff_doc[] = "";
static char module_doc[] = "";

static PyMethodDef cfunc_methods[] = {
	{"diff", diff, METH_VARARGS, diff_doc},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcfunc() {
	Py_InitModule3("cfunc", cfunc_methods, module_doc);
	import_array();
}
