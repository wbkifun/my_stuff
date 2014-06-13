#include <stdio.h>

int main()
{
	int i;
	double *A;

	A = (double *) malloc(sizeof(double)*N);

#pragma omp parallel for
	for(i=0; i<N; i++)
		A[i] = 0.0;

#pragma omp parallel for
	for(i=0; i<N; i++) {
		// ......
	}
}
