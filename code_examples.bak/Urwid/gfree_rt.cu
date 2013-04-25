#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char* argv[]) {
	int ngpu;
	int tid;
	size_t *frees;
	size_t *totals;

	cudaGetDeviceCount(&ngpu);
	//printf("ngpu = %d\n", ngpu);
	frees = (size_t *) calloc (ngpu, sizeof(size_t));
	totals = (size_t *) calloc (ngpu, sizeof(size_t));

	omp_set_num_threads(ngpu);
	#pragma omp parallel private(tid) shared(frees, totals)
	{
		tid = omp_get_thread_num();
		//printf("nthreads = %d, tid = %d\n", omp_get_num_threads(), tid);
		cudaSetDevice(tid);
		cudaMemGetInfo((size_t*)&frees[tid], (size_t*)&totals[tid]);
	}

	printf ("\ttotal\t\tfree\t\tused\n");
	for(int i=0; i<ngpu; i++) {
		printf("GPU %d\t%lu\t%lu\t%lu\n", i, (size_t)totals[i], (size_t)frees[i], (size_t)totals[i]-(size_t)frees[i]);
	}

	return 0;
}
