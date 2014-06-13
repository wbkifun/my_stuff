#include <stdio.h>
#include <omp.h>
#define N 4

int main()
{
	int i, tid;
	omp_set_num_threads(4);
#pragma omp parallel private(i, tid)
{
	tid = omp_get_thread_num();
	#pragma omp sections
	{
		#pragma omp section
		{
			for(i=0; i<N; i++)
				printf("L1 tid=%d\n", tid);
		}
		#pragma omp section
		{
			for(i=0; i<N; i++)
				printf("L2 tid=%d\n", tid);
			sleep(2);
		}
	}
	printf("end tid=%d\n", tid);
} // pragma omp parallel
}
