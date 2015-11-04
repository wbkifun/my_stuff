#include <stdlib.h>
#include <stdio.h>
#include <omp.h>



void daxpy(int n, double a, double *x, double *y) {
	int i;

	#pragma omp parallel for simd shared(x,y) schedule(static)
	for (i = 0; i<n; i++) {
		y[i] = a*x[i] + y[i];
	}
}


int main(void) {
	omp_set_num_threads(1);

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
