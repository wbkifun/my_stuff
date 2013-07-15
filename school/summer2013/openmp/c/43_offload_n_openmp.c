#include <stdio.h>

int main()
{
	long i, sum=0;
#pragma omp parallel
{
	#pragma omp single
	#pragma offload target(mic)
	{
		printf("check\n");
	}
	#pragma omp for reduction(+:sum)
	for(i=1; i<=N; i++)
		sum += i;
}
	printf("sum = %ld\n", sum);
}
