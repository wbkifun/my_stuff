#include <stdio.h>
#include <omp.h>

int main()
{
#pragma offload target(mic:0)
#pragma omp parallel
	{
		printf("tid = %d threads = %d\n",
				omp_get_thread_num(), omp_get_num_threads() );
	}
#pragma offload target(mic:1)
#pragma omp parallel
	{
		printf("tid = %d threads = %d\n",
				omp_get_thread_num(), omp_get_num_threads() );
	}
}
