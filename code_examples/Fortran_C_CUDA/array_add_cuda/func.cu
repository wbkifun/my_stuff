#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void add_gpu(int nn, double *a, double *b, double *c) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < nn) {
		c[tid] = a[tid] + b[tid];
	}
}




extern "C" void add(int nx, int ny, void *ap, void *bp, void *cp) { 
	int i, j, idx;
	
	// arrays in the GPU
	static int call_count=1; 
	static double *a_gpu, *b_gpu, *c_gpu;
	if (call_count == 1) {
		printf("******************** call count %d ********************\n", call_count);
		cudaMalloc((void**)&a_gpu, nx*ny*sizeof(double));
		cudaMalloc((void**)&b_gpu, nx*ny*sizeof(double));
		cudaMalloc((void**)&c_gpu, nx*ny*sizeof(double));
	}


	// arrays from Fortran
	double *a, *b, *c;
	a = (double *)ap;
	b = (double *)bp;
	c = (double *)cp;
	printf("#C address: a %p, b %p, c %p\n", a, b, c);


	cudaMemcpy(a_gpu, a, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b, nx*ny*sizeof(double), cudaMemcpyHostToDevice);


	// call the kernel
	add_gpu<<<1,nx*ny>>>(nx*ny, a_gpu, b_gpu, c_gpu);


	// copy array 'c' back from the GPU
	cudaMemcpy(c, c_gpu, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);

	
	// print result
	for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
			idx = i*ny + j;
			printf("c[%d,%d]= %g\n", i, j, c[idx]);
		}
	}

	call_count += 1;
}
