#include <stdio.h>
#include <omp.h>

int main()
{
	long i, sum=0;

#pragma omp parallel
{
	#pragma omp for
	for(i=1; i<=N; i++)
		#pragma omp critical
		sum += i;
}

	printf("sum = %ld\n", sum);
}
