#include <stdio.h>

int main()
{
#pragma omp parallel
{
	#pragma omp sections
	{
		#pragma omp section
		{
			#pragma offload target(mic:0)
			#pragma omp parallel
			{
				printf("hello %d\n", omp_get_thread_num());
			}
		}
		#pragma omp section
		{
			#pragma offload target(mic:1)
			#pragma omp parallel
			{
				printf("hello %d\n", omp_get_thread_num());
			}
		}
	}
}
}
