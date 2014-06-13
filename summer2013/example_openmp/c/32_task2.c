#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp parallel num_threads(32)
{
	#pragma omp single
	{
		printf("A tid=%d\n", omp_get_thread_num() );
		printf("B tid=%d\n", omp_get_thread_num() );
		printf("C tid=%d\n", omp_get_thread_num() );
		printf("D tid=%d\n", omp_get_thread_num() );
	}
}
}
