#include <python2.7/Python.h>
#include <omp.h>
#include <xmmintrin.h>
#define LOADU _mm_loadu_ps // not aligned to 16 bytes
#define LOAD _mm_load_ps	
#define STORE _mm_store_ps
#define ADD _mm_add_ps


void vecadd(int nx, float *a, float *b, float *c) {
	int idx;
	__m128 xa, xb;

    //Py_Initialize();
    //PyEval_InitThreads();
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for private(idx, xa, xb)
	for( idx=0; idx<nx; idx+=4 ) {
        xa = LOAD(a+idx);
        xb = LOAD(b+idx);
        STORE(c+idx, ADD(xa, xb));
	}
    Py_END_ALLOW_THREADS
    //Py_Finalize();
}
