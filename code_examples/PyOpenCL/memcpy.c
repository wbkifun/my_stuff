#include <xmmintrin.h>
#include <omp.h>
#define LOAD _mm_load_ps	
#define STORE _mm_store_ps
#define SET _mm_set_ps1
//#define OMP_MAX_THREADS 4


void memcpy_htoh(int nx, float *a, float *b) {
	int idx;

	omp_set_num_threads(OMP_MAX_THREADS);
	#pragma omp parallel for \
	shared(a, b) \
	private(idx) \
	schedule(guided)
	for ( idx=0; idx<nx; idx+=4 ) {
		STORE(b+idx, LOAD(a+idx));
	}
}


void memread(int nx, float *a) {
	int idx;
	__m128 ch;

	omp_set_num_threads(OMP_MAX_THREADS);
	#pragma omp parallel for \
	shared(a, ch) \
	private(idx) \
	schedule(guided)
	for ( idx=0; idx<nx; idx+=4 ) {
		ch = LOAD(a+idx);
	}
}


void memwrite(int nx, float *a) {
	int idx;

	omp_set_num_threads(OMP_MAX_THREADS);
	#pragma omp parallel for \
	shared(a) \
	private(idx) \
	schedule(guided)
	for ( idx=0; idx<nx; idx+=4 ) {
		STORE(a, SET(idx*0.01));
	}
}
