#include <stdio.h>
#include <omp.h>

int main()
{
	long i, sum=0;

#pragma omp parallel for reduction(+:sum)
	for(i=1; i<=N; i++)
		sum += i;

	printf("sum = %ld\n", sum);
}
