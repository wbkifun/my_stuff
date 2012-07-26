#include <stdio.h>
#include <stdlib.h>


__global__ void doubling(int n, float *a) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	int i;

	while(tid < n) {
		a[tid] *= 2;
		for(i=0; i<1000; i++) a[tid] *= 1;
		tid += blockDim.x * gridDim.x;
	}
}


int main() {
	int i;
	int n=65535*3500;

	float *a, *a_dev;
	a = (float *)malloc(n*sizeof(float));
	cudaMalloc((void**)&a_dev, n*sizeof(float));

	for(i=0; i<n; i++) a[i] = 1.; 
	cudaMemcpy(a_dev, a, n*sizeof(float), cudaMemcpyHostToDevice);

	doubling<<<65535, 256>>>(n, a_dev);
	cudaMemcpy(a, a_dev, n*sizeof(float), cudaMemcpyDeviceToHost);

	for(i=0; i<n; i++) {
		if(abs(a[i] - 2.) > 1e-5) printf("a[%d] = %g\n", i, a[i]); 
	}

	free(a);
	cudaFree(a_dev);
	return 0;
}
