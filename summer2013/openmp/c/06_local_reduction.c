#include <stdio.h>
#include <omp.h>

int main()
{
	long i, sum=0, local_sum;

#pragma omp parallel private(i, local_sum)
{
	local_sum = 0;
	#pragma omp for
	for(i=1; i<=N; i++) 
		local_sum += i;

	#pragma omp atomic
	sum += local_sum;
}

	printf("sum = %ld\n", sum);
}
