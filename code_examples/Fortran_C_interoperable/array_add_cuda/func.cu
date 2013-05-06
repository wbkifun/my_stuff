#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void add_gpu(int nn, double *aa, double *bb, double *cc) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < nn) {
		cc[tid] = aa[tid] + bb[tid];
	}
}




extern "C" void add_(int nx, int ny, void *ap, void *bp, void *cp) { 
	static int round = 1; 
	int i, j, idx;
	

	// arrays from Fortran
	double *aa, *bb, *cc;
	aa = (double *)ap;
	bb = (double *)bp;
	cc = (double *)cp;
	printf("#C address: aa %p, bb %p, cc %p\n", aa, bb, cc);


	// arrays in the GPU
	static double *a_gpu, *b_gpu, *c_gpu;
	if (round == 1);
		cudaMalloc((void**)&a_gpu, nx*ny*sizeof(double));
		cudaMalloc((void**)&b_gpu, nx*ny*sizeof(double));
		cudaMalloc((void**)&c_gpu, nx*ny*sizeof(double));

	cudaMemcpy(a_gpu, aa, nx*ny*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, bb, nx*ny*sizeof(double), cudaMemcpyHostToDevice);


	// call the kernel
	add_gpu<<<1,nx*ny>>>(nx*ny, a_gpu, b_gpu, c_gpu);


	// copy array 'c' back from the GPU
	cudaMemcpy(cc, c_gpu, nx*ny*sizeof(double), cudaMemcpyDeviceToHost);

	
	if ( round == 1)
		printf("*********************************************\n");

	for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
			idx = i*ny + j;
			cc[idx] = aa[idx] + bb[idx];
			printf("cc[%d,%d]= %g\n", i, j, cc[idx]);
		}
	}

	round += 1;
}
