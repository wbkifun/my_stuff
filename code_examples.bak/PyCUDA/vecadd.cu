#include <stdio.h>
#include <stdlib.h>

void vecadd(int nx, float *a, float *b, float *c) {
	int i;
	for(i=0; i<nx; i++) c[i] = a[i] + b[i];
}


__global__ void vecadd_gpu(int nx, float *a_gpu, float *b_gpu, float *c_gpu) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<nx) c_gpu[i] = a_gpu[i] + b_gpu[i];
}


int main() {
	int i, nx=1000;

	// allocate arrays
	float *a, *b, *c, *c2;
	a = (float *) malloc(nx*sizeof(float));
	b = (float *) malloc(nx*sizeof(float));
	c = (float *) calloc(nx,sizeof(float));
	c2 = (float *) calloc(nx,sizeof(float));

	// initialize a, b arrays
	for(i=0; i<nx; i++) {
		a[i] = i;
		b[i] = 2*i;
	}

	// allocate device arrays
	float *a_gpu, *b_gpu, *c_gpu;
	cudaMalloc((void**)&a_gpu, nx*sizeof(float));
	cudaMalloc((void**)&b_gpu, nx*sizeof(float));
	cudaMalloc((void**)&c_gpu, nx*sizeof(float));

	// copy arrays host to device
	cudaMemcpy(a_gpu, a, nx*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b, nx*sizeof(float), cudaMemcpyHostToDevice);

	// vector add
	vecadd(nx, a, b, c);
	vecadd_gpu<<<dim3(nx/256+1,1),dim3(256,1,1)>>>(nx, a_gpu, b_gpu, c_gpu);

	// copy arrays device to host
	cudaMemcpy(c2, c_gpu, nx*sizeof(float), cudaMemcpyDeviceToHost);

	// print
	for(i=0; i<nx; i++) printf("c[%d]-c_gpu[%d]= %g\n", i, i, c[i]-c2[i]);
	
	return 0;
}
