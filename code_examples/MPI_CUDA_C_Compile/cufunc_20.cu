extern "C" {
	#include "20.h"
}

__global__ void cusum( int rank, float *data ) {
	int i = threadIdx.x;
	data[i] += rank;
}


__host__ void cufunc( int rank, float *data ) {
	float *dev_data;
	int array_size = CHUNKSIZE*sizeof(float);

	cudaMalloc ( (void**) &dev_data, array_size );

	cudaMemcpy( dev_data, data, array_size, cudaMemcpyHostToDevice );
	cusum <<<dim3(1),dim3(CHUNKSIZE)>>> ( rank, dev_data );
	cudaMemcpy( data, dev_data, array_size, cudaMemcpyDeviceToHost );
}
