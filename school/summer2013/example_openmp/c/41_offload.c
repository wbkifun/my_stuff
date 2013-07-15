#include <stdio.h>

int main()
{
	long i, sum=0;
#pragma offload target(mic)
	#pragma omp parallel for reduction(+:sum)
	for(i=1; i<=N; i++)
		sum += i;

	printf("sum = %ld\n", sum);
}
