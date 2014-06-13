#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel num_threads(32)
{
	#pragma omp single
	{
		printf("A tid=%d\n", omp_get_thread_num() );
		#pragma omp task
		{
			printf("B tid=%d\n", omp_get_thread_num() );
		}
		#pragma omp task
		{
			printf("C tid=%d\n", omp_get_thread_num() );
		}
		#pragma omp taskwait
		printf("D tid=%d\n", omp_get_thread_num() );
	}
}
}
