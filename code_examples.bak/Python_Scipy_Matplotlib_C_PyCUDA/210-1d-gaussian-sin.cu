#include <math.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void mul_array(int n, float *a, float *b, float *c) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n) c[i] = a[i]*b[i];
}

int main() {
	int i, n=1000;
	float x;
	int bytes=n*sizeof(float);
	float *a = (float*)malloc(bytes);
	float *b = (float*)malloc(bytes);
	float *c = (float*)malloc(bytes);

	for(i=0; i<n; i++) {
		x = -5+0.01*i;
		a[i] = exp(-x*x/2);
		b[i] = sin(5*x);
	}

	for(i=0; i<n; i++) c[i] = a[i]*b[i];

	float *c2 = (float*)malloc(bytes);

	float *a_gpu, *b_gpu, *c_gpu;
	cudaMalloc((void**)&a_gpu, bytes);
	cudaMalloc((void**)&b_gpu, bytes);
	cudaMalloc((void**)&c_gpu, bytes);

	cudaMemcpy(a_gpu, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b, bytes, cudaMemcpyHostToDevice);

	dim3 Db = dim3(256,1,1);		// threads per block
	dim3 Dg = dim3(n/256+1,1);		// threads per block
	mul_array<<<Dg,Db>>>(n, a_gpu, b_gpu, c_gpu);

	cudaMemcpy(c2, c_gpu, bytes, cudaMemcpyDeviceToHost);

	for(i=0; i<n; i++) printf("%g, %g, %g\n",c[i], c2[i], c[i]-c2[i] );

	free(a); free(b); free(c);
	cudaFree(a_gpu); cudaFree(b_gpu); cudaFree(c_gpu);
}
