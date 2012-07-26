#include <Python.h>
#include <numpy/arrayobject.h>

#include <xmmintrin.h>
#define LOADU _mm_loadu_ps	// not aligned to 16 bytes
#define LOAD _mm_load_ps	
#define STORE _mm_store_ps
#define ADD _mm_add_ps
#define SUB _mm_sub_ps
#define MUL _mm_mul_ps


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

	__m128 vc, vf, vg, vg1, vg2, vg3, vg4, tmp;
	__m128 c2 = {2,2,2,2}, c4 = {4,4,4,4};
	#pragma omp parallel for \
	shared(ny, c, f, g, c2, c4) \
	private(vc, vf, vg, vg1, vg2, vg3, vg4, tmp, i, j, idx) \
	schedule(guided)
	for ( i=1; i<nx-1; i++ ) {
		for ( j=0; j<ny; j+=4 ) {
			idx = i*ny + j;
			vc = LOAD(c+idx);
			vf = LOAD(f+idx);
			vg = LOAD(g+idx);
			vg1 = LOAD(g+idx+ny);
			vg2 = LOAD(g+idx-ny);
			vg3 = LOADU(g+idx+1);
			vg4 = LOADU(g+idx-1);
			tmp = ADD(ADD(ADD(vg1,vg2),vg3),vg4);
			tmp = MUL(vc,SUB(tmp,MUL(c4,vg)));
			tmp = SUB(ADD(tmp,MUL(c2,vg)),vf);
			STORE(f+idx,tmp);
		}
	}

   	Py_INCREF(Py_None);
   	return Py_None;
}

/* =============================================================
 * method table listing
 * module's initialization
============================================================= */
/*
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
*/
static PyMethodDef cfunc_methods[] = {
	{"update", update, METH_VARARGS, ""},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initwave2d_cfunc() {
	Py_InitModule3("wave2d_cfunc", cfunc_methods, "");
	import_array();
}
