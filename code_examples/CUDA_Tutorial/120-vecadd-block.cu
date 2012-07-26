#include <stdio.h>
#include <stdlib.h>

void print_array(int n, char str, float *a) {
	printf("%c:  ", str);
	for(int i=0; i<n; i++) printf("\t%f", a[i]);
	printf("\n");
}

void vecadd(int n, float *a, float *b, float *c) {
	for(int i=0; i<n; i++) {
		c[i] = a[i] + b[i];
	}
}

__global__ void vecadd_gpu(int n, float *a, float *b, float *c, int offset) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x + offset;

	if( tid<n ) {
		c[tid] = a[tid] + b[tid];
	}
}


int main() {
	int i;
	int n=300000000;
	float *a, *b, *c;

	// allocation in host memory
	a = (float *)malloc(n*sizeof(float));
	b = (float *)malloc(n*sizeof(float));
	c = (float *)malloc(n*sizeof(float));

	// initialize
	for(i=0; i<n; i++) {
		//a[i] = i+5.5;
		//b[i] = -1.2*i;
		a[i] = rand()/(RAND_MAX+1.);
		b[i] = rand()/(RAND_MAX+1.);
	}
	//print_array(n, 'a', a);
	//print_array(n, 'b', b);

	// call the function
	vecadd(n, a, b, c);
	//printf("results from CPU\n");
	//print_array(n, 'c', c);

	// allocation in device memory
	float *a_dev, *b_dev, *c_dev;
	cudaMalloc((void**)&a_dev, n*sizeof(float));
	cudaMalloc((void**)&b_dev, n*sizeof(float));
	cudaMalloc((void**)&c_dev, n*sizeof(float));
	
	// copy arrays 'a' and 'b' to the device
	cudaMemcpy(a_dev, a, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, b, n*sizeof(float), cudaMemcpyHostToDevice);
	
	// call the kernel
	int tpb =256;	// thread per block
    int max_bpg = 65535;
	int ng = n/(max_bpg*tpb); 	// number of grid
	for(i=0; i<ng; i++) {
		vecadd_gpu<<<max_bpg,tpb>>>(n, a_dev, b_dev, c_dev, i*max_bpg*tpb);
	}
	if( n%(max_bpg*tpb)!=0 ) {
		int nn = n-ng*max_bpg*tpb;
		int bpg = nn%tpb==0 ? nn/tpb : nn/tpb+1;
		vecadd_gpu<<<bpg,tpb>>>(n, a_dev, b_dev, c_dev, ng*max_bpg*tpb);
	}
			
	// copy array 'c' back from the device to the host
	float *c2;
	c2 = (float *)malloc(n*sizeof(float));
	cudaMemcpy(c2, c_dev, n*sizeof(float), cudaMemcpyDeviceToHost);
	//printf("results from GPU\n");
	//print_array(n, 'c', c2);

	printf("n=%d\n", n);
	printf("Check results..");
	float diff;
	for(i=0; i<n; i++) {
		diff = fabs(c2[i]-c[i]);
		if(diff > 1e-7) break;
	}
	if(diff > 1e-7) printf("Mismatch!\n");
	else printf("OK!\n");

	free(a);
	free(b);
	free(c);
	free(c2);
	cudaFree(a_dev);
	cudaFree(b_dev);
	cudaFree(c_dev);
	return 0;
}
