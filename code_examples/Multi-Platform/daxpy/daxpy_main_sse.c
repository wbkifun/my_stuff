#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <emmintrin.h>
#define LOADU _mm_loadu_pd	// not aligned to 16 bytes
#define LOAD _mm_load_pd
#define STORE _mm_store_pd
#define ADD _mm_add_pd
#define SUB _mm_sub_pd
#define MUL _mm_mul_pd



void daxpy(int n, double a, double *x, double *y) {
	int i;
	__m128d vx, vy;
	__m128d a2 = {a,a};

	#pragma omp parallel for shared(x,y) schedule(static)
	for (i = 0; i<n; i+=2) {
		vx = LOAD(x+i);
		vy = LOAD(y+i);

		STORE(y+i,ADD(MUL(a2,vx),vy));
	}
}


int main(void) {
	//omp_set_num_threads(1);

	int nx=100000000;
	double a=2.7;
	double *x, *y;

	int i;

	x = (double *)malloc(nx*sizeof(double));
	y = (double *)malloc(nx*sizeof(double));

	for (i=0; i<nx; i++) {
		x[i] = i/2.;
		y[i] = i/3.;
	}

	for (i=0; i<1000; i++)
		daxpy(nx, a, x, y);

	//printf("y[1] = 2.7*0.5+0.33333 = 1.68333 = %.5f\n",y[1]);
	printf("y[1] = 1350.3333, diff = %g\n",y[1]-1350.3333333);

	return 0;
}
