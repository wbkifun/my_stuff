#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <xmmintrin.h>
#define LOADU _mm_loadu_ps // not aligned to 16 bytes
#define LOAD _mm_load_ps	
#define STORE _mm_store_ps
#define ADD _mm_add_ps
#define SUB _mm_sub_ps
#define MUL _mm_mul_ps


static PyObject *vecadd(PyObject *self, PyObject *args) {
    PyArrayObject *A, *B, *C;
    if (!PyArg_ParseTuple(args, "OOO", &A, &B, &C)) return NULL;

	int nx, idx;
    float *a, *b, *c;
    nx = (int)(A->dimensions)[0];
    a = (float*)(A->data);
    b = (float*)(B->data);
    c = (float*)(C->data);

	__m128 xa, xb;

    Py_BEGIN_ALLOW_THREADS
    //omp_set_num_threads(0);
    #pragma omp parallel for private(idx, xa, xb)
	for( idx=0; idx<nx; idx+=4 ) {
        xa = LOAD(a+idx);
        xb = LOAD(b+idx);
        STORE(c+idx, ADD(xa, xb));
	}
    Py_END_ALLOW_THREADS
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *vecsub(PyObject *self, PyObject *args) {
    PyArrayObject *A, *B, *C;
    if (!PyArg_ParseTuple(args, "OOO", &A, &B, &C)) return NULL;

	int nx, idx;
    float *a, *b, *c;
    nx = (int)(A->dimensions)[0];
    a = (float*)(A->data);
    b = (float*)(B->data);
    c = (float*)(C->data);

	__m128 xa, xb;

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for private(idx, xa, xb)
	for( idx=0; idx<nx; idx+=4 ) {
        xa = LOAD(a+idx);
        xb = LOAD(b+idx);
        STORE(c+idx, SUB(xa, xb));
	}
    Py_END_ALLOW_THREADS
    Py_INCREF(Py_None);
    return Py_None;
}



static PyMethodDef ufunc_methods[] = {
    {"vecadd", vecadd, METH_VARARGS, ""},
    {"vecsub", vecsub, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initvecop() {
    Py_InitModule("vecop", ufunc_methods);
    import_array();
}
