#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>


int main() {

	int ngpu;
	CUdevice cuDevice;
	CUcontext cuContext;
	cuInit(0);
	cuDeviceGetCount(&ngpu);
	//printf("ngpu = %d\n", ngpu);

	size_t *totals, *frees ;
	totals = (size_t *) calloc (ngpu, sizeof(size_t));
	frees = (size_t *) calloc (ngpu, sizeof(size_t));

	int tid;
	omp_set_num_threads(ngpu);
	#pragma omp parallel private(tid, cuDevice, cuContext) shared(frees, totals)
	{
		tid = omp_get_thread_num();
		//printf("nthreads = %d, tid = %d\n", omp_get_num_threads(), tid);
		cuDeviceGet(&cuDevice, tid);
		cuCtxCreate(&cuContext, tid, cuDevice);
		cuMemGetInfo((size_t*)&frees[tid], (size_t*)&totals[tid]);
	}

	printf ("\ttotal\t\tfree\t\tused\n");
	for(int i=0; i<ngpu; i++) {
		printf("GPU %d\t%lu\t%lu\t%lu\n", i, (size_t)totals[i], (size_t)frees[i], (size_t)totals[i]-(size_t)frees[i]);
	}

	return 0;
}
