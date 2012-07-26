#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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


int main(int argc, char **argv) {
	int count, rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;

	cudaGetDeviceCount(&count);
	if( rank==0 ) {
		printf("Number of CUDA-enabled GPU devices: %d\n", count);
		printf("MPI size: %d\n", size);
	}

	int i, n=300000000;
	float *a, *b;

	a = (float *)malloc(n*sizeof(float));
	b = (float *)malloc(n*sizeof(float));

	for(i=0; i<n; i++) {
		a[i] = rand()/(RAND_MAX+1.);
		b[i] = rand()/(RAND_MAX+1.);
	}

	if( rank==0 ) {
		float *c, *c2;
		c = (float *)malloc(n*sizeof(float));
		c2 = (float *)malloc(n*sizeof(float));
		vecadd(n, a, b, c);

		printf("rank=%d, n=%d\n", rank, n);
		for(i=1; i<size; i++) {
			MPI_Recv(&c2[n/count*(i-1)], n/count, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
		}

		printf("rank=%d, n=%d\n", rank, n);
		printf("n=%d\n", n);
		printf("Check results..");
		float diff;
		for(i=0; i<n; i++) {
			diff = fabs(c2[i]-c[i]);
			if(diff > 1e-7) break;
		}
		if(diff > 1e-7) printf("Mismatch!\n");
		else printf("OK!\n");
	}
	else {
		cudaSetDevice(rank-1);
		printf("rank=%d, n=%d\n", rank, n);
		n = n/count;
		printf("rank=%d, n=%d\n", rank, n);

		printf("rank=%d, 1\n", rank);
		float *a_dev, *b_dev, *c_dev;
		cudaMalloc((void**)&a_dev, n*sizeof(float));
		cudaMalloc((void**)&b_dev, n*sizeof(float));
		cudaMalloc((void**)&c_dev, n*sizeof(float));
		printf("rank=%d, 2\n", rank);

		cudaMemcpy(a_dev, &a[n*(rank-1)], n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(b_dev, &b[n*(rank-1)], n*sizeof(float), cudaMemcpyHostToDevice);

		printf("rank=%d, 3\n", rank);
		int tpb =256;	// thread per block
		int max_bpg = 65535;
		int ng = n/(max_bpg*tpb); 	// number of grid
		for(i=0; i<ng; i++) {
			vecadd_gpu<<<max_bpg,tpb>>>(n, a_dev, b_dev, c_dev, i*max_bpg*tpb);
		}
		printf("rank=%d, 4\n", rank);
		if( n%(max_bpg*tpb)!=0 ) {
			int nn = n-ng*max_bpg*tpb;
			int bpg = nn%tpb==0 ? nn/tpb : nn/tpb+1;
			vecadd_gpu<<<bpg,tpb>>>(n, a_dev, b_dev, c_dev, ng*max_bpg*tpb);
		}
		printf("rank=%d, 5\n", rank);

		float *c2;
		c2 = (float *)malloc(n*sizeof(float));
		cudaMemcpy(c2, c_dev, n*sizeof(float), cudaMemcpyDeviceToHost);
		MPI_Send(c2, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}
