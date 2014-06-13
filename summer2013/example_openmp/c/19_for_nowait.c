#include <stdio.h>
#define N 20000

int a[N][N];

int main()
{
	int i, j;

#pragma omp parallel private(i,j)
{
	#pragma omp for nowait
	for(i=0; i<N; i++)
		for(j=i; j<N; j++)
			a[i][j] = i+j;

	#pragma omp for
	for(i=0; i<N; i++)
		for(j=0; j<i; j++)
			a[i][j] = i-j;
} // pragma omp parallel
}
