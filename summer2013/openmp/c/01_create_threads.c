#include <stdio.h>
#include <omp.h>

int main()
{
	printf("threads = %d\n", omp_get_num_threads() );
#pragma omp parallel num_threads(3)
	{
		printf("tid = %d threads = %d\n",
				omp_get_thread_num(), omp_get_num_threads() );
	}
	printf("threads = %d\n", omp_get_num_threads() );
#pragma omp parallel
	{
		printf("tid = %d threads = %d\n",
				omp_get_thread_num(), omp_get_num_threads() );
	}
	omp_set_num_threads(4);
	printf("threads = %d\n", omp_get_num_threads() );
#pragma omp parallel
	{
		printf("tid = %d threads = %d\n",
				omp_get_thread_num(), omp_get_num_threads() );
	}
	printf("threads = %d\n", omp_get_num_threads() );
}
