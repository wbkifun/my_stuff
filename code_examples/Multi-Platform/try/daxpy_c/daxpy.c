#include <math.h>



void daxpy(int n, double a, double *x, double *y) {
	int i;

	for (i = 0; i<n; i++) {
		y[i] = a*x[i] + y[i];
	}
}
