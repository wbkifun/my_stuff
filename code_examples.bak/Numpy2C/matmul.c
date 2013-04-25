#include <Python.h>
//#include <Numeric/arrayobject.h>
#include <numpy/arrayobject.h>
#define P1F(a, i) *((float *)(a->data + (i)*a->strides[0]))
#define P2F(a, i, j) *((float *)(a->data + (i)*a->strides[0] + (j)*a->strides[1]))

static PyObject *matmul(PyObject *self, PyObject *args) {
	int N;
	PyArrayObject *A, *B, *C;

	if (!PyArg_ParseTuple(args, "iOOO",
								&N, &A, &B, &C)) {
		return NULL;
	}

	printf("A->nd = %d\n",A->nd);
	printf("A->descr->type_num = %d\n", A->descr->type_num);
	printf("PyArray_DOUBLE = %d\n", PyArray_DOUBLE);
	printf("PyArray_FLOAT = %d\n", PyArray_FLOAT);

	if (A->nd != 2 || A->descr->type_num != PyArray_FLOAT) {
		PyErr_Format(PyExc_ValueError,"A array is %d-dimensional or not of type double", A->nd);
		return NULL;
	}
	
	float stmp;
	int i,j,k;

	printf("N=%d\n",N);
	
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			stmp=0.;
			for (k=0; k<N; k++) {
				stmp += P2F(A,i,k)*P2F(B,k,j);
				P2F(C,i,j) = stmp;
			}
		}
	}
	
	Py_INCREF(Py_None);
	return Py_None;
}

static char matmul_doc[] = "matmul(N, A, B, C)";
static char module_doc[] = \
	"module ext_matmul:\n\
	matmul(N, A, B, C)";

static PyMethodDef matmul_methods[] = {
	{"matmul", matmul, METH_VARARGS, matmul_doc},
};

PyMODINIT_FUNC initext_matmul() {
	Py_InitModule3("ext_matmul", matmul_methods, module_doc);
	import_array();
}
