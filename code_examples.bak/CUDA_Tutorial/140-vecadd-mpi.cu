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
	//MPI_Status status;

	cudaGetDeviceCount(&count);
	if( rank==0 ) {
		printf("MPI size: %d\n", size);
		printf("Number of CUDA-enabled GPU devices: %d\n", count);
	}
	cudaSetDevice(rank);

	int i;
	int tn=600000000;
	int n = tn/count;

	float *ta, *tb, *tc;
	if( rank==0 ) {
		ta = (float *)malloc(tn*sizeof(float));
		tb = (float *)malloc(tn*sizeof(float));
		tc = (float *)malloc(tn*sizeof(float));

		for(i=0; i<tn; i++) {
			ta[i] = rand()/(RAND_MAX+1.);
			tb[i] = rand()/(RAND_MAX+1.);
		}
	}

	float *a, *b;
	a = (float *)malloc(n*sizeof(float));
	b = (float *)malloc(n*sizeof(float));

	MPI_Scatter(ta, n, MPI_FLOAT, a, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatter(tb, n, MPI_FLOAT, b, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

	float *a_dev, *b_dev, *c_dev;
	cudaMalloc((void**)&a_dev, n*sizeof(float));
	cudaMalloc((void**)&b_dev, n*sizeof(float));
	cudaMalloc((void**)&c_dev, n*sizeof(float));

	cudaMemcpy(a_dev, a, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, b, n*sizeof(float), cudaMemcpyHostToDevice);

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

	if( rank==0 ) vecadd(tn, ta, tb, tc);

	cudaMemcpy(b, c_dev, n*sizeof(float), cudaMemcpyDeviceToHost);
	MPI_Gather(b, n, MPI_FLOAT, tb, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if( rank==0 ) {
		printf("tn=%d, n=%d\n", tn, n);
		printf("Check results..");
		float diff;
		for(i=0; i<tn; i++) {
			diff = fabs(tc[i]-tb[i]);
			if(diff > 1e-7) break;
		}
		if(diff > 1e-7) printf("Mismatch!\n");
		else printf("OK!\n");
	}

	MPI_Finalize();
	return 0;
}
