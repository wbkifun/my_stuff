#include <stdio.h>

int main()
{
	long i, sum=0, local_sum;
#pragma omp parallel private(local_sum)
{
	local_sum = 0;
	#pragma omp single nowait
	#pragma offload target(mic)
	{
		printf("check\n");
	}
	#pragma omp for schedule(dynamic)
	for(i=1; i<=N; i++)
		local_sum += i;

	#pragma omp atomic
	sum += local_sum;
}
	printf("sum = %ld\n", sum);
}
