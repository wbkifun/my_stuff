#include <Python.h>
//#include <Numeric/arrayobject.h>
#include <numpy/arrayobject.h>
#define P1F(a, i) *((float *)(a->data + (i)*a->strides[0]))
#define P2F(a, i, j) *((float *)(a->data + (i)*a->strides[0] + (j)*a->strides[1]))

static PyObject *arr1d(PyObject *self, PyObject *args) {
	int N;
	PyArrayObject *A, *B;

	if (!PyArg_ParseTuple(args, "iOO",
				&N, &A, &B )) {
		return NULL;
	}

	printf("A->nd = %d\n",A->nd);
	printf("A->descr->type_num = %d\n", A->descr->type_num);
	printf("PyArray_DOUBLE = %d\n", PyArray_DOUBLE);
	printf("PyArray_FLOAT = %d\n", PyArray_FLOAT);

	int i;

	printf("N=%d\n",N);
	
	for (i=0; i<N; i++) {
		((float *)(B->data))[i] = ((float *)(A->data))[i]*2;
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}

static char arr1d_doc[] = "arr1d(N, A, B)";
static char module_doc[] = \
	"module ext_arr1d:\n\
	arr1d(N, A, B)";

static PyMethodDef arr1d_methods[] = {
	{"arr1d", arr1d, METH_VARARGS, arr1d_doc},
};

PyMODINIT_FUNC initext_arr1d() {
	Py_InitModule3("ext_arr1d", arr1d_methods, module_doc);
	import_array();
}
