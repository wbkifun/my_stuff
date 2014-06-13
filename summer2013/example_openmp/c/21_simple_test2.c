#include <stdio.h>
#include <omp.h>
#define N 20

int main()
{
	int i, a[N];

	omp_set_num_threads(4);
#pragma omp parallel private(i)
{
	if( omp_get_thread_num() != 2 )
		sleep(2);
	#pragma omp for nowait
	for(i=0; i<N; i++) {
		a[i] = i;
		printf("a[%d]=%d tid=%d\n", i, a[i], omp_get_thread_num());
	}
	printf("end %d thread\n", omp_get_thread_num());
} // pragma omp parallel private(i)
}

