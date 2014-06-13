#include <stdio.h>
#include <omp.h>

int main()
{
	long i, sum=0, tid, local_sum[TN*8];

#pragma omp parallel private(tid) num_threads(TN)
{
	tid = omp_get_thread_num();
	local_sum[tid*8] = 0;

	#pragma omp for
	for(i=1; i<=N; i++)
		local_sum[tid*8] += i;
}
	for(i=0; i<TN; i++)
		sum += local_sum[i*8];

	printf("sum = %ld\n", sum);
}
