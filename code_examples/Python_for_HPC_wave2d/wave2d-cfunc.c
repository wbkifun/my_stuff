/* ----------------------------------------------------------------------------
# File Name : wave2d_cfunc.c
#
# Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
# 
# Written date :	2010. 6. 17
# Modify date :		
#
# Copyright : GNU GPL
#
# Description : 
# Simulation for the 2-dimensional wave equations with simple FD (Finite-Difference) scheme
#
# These are educational codes to study python programming for high performance computing.
# Step 1: Using numpy arrays
# Step 2: Combining with C function
---------------------------------------------------------------------------- */

#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *update(PyObject *self, PyObject *args) {
	PyArrayObject *C, *F, *G;
	if (!PyArg_ParseTuple(args, "OOO", &C, &F, &G )) return NULL;

	int nx, ny, i, j, idx;
	float *c, *f, *g;
	nx = (int)(C->dimensions)[0];
	ny = (int)(C->dimensions)[1];
	c = (float*)(C->data);
	f = (float*)(F->data);
	g = (float*)(G->data);

	for ( i=1; i<nx-1; i++ ) {
		for ( j=1; j<ny-1; j++ ) {
			idx = i*ny + j;
			f[idx] = c[idx]*(g[idx+ny] + g[idx-ny] + g[idx+1] + g[idx-1] - 4*g[idx]) + 2*g[idx] - f[idx];
		}
	}

   	Py_INCREF(Py_None);
   	return Py_None;
}

/* -------------------------------------------------------------
# Method table listing
# Module's initialization
------------------------------------------------------------- */
static char update_doc[] = "";
static char module_doc[] = "";

static PyMethodDef wave2d_cfunc_methods[] = {
	{"update", update, METH_VARARGS, update_doc},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initwave2d_cfunc() {
	Py_InitModule3("wave2d_cfunc", wave2d_cfunc_methods, module_doc);
	import_array();
}
