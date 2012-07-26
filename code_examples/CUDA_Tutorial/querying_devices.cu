#include <stdio.h>

int main() {
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	printf("Number of CUDA-enabled GPU devices: %d\n", count);
	for(int i=0; i<count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("Device %d: \"%s\"\n", i, prop.name);
		printf("  Compute Capability:                   %d.%d\n", prop.major, prop.minor);
		printf("  Number of multi processor:            %d\n", prop.multiProcessorCount);
		printf("  Number of scalar processor:           %d\n", prop.multiProcessorCount*8);
		printf("  Clock rate:                           %d MHz\n", prop.clockRate/(1024));
		printf("  Total amount of global memory:        %1.2f GBytes\n", (float)(prop.totalGlobalMem)/(1024*1024*1024));
		printf("  Total amount of shared memory per MP: %d KBytes\n", prop.regsPerBlock/1024);
		printf("  Threads in a warp:                    %d\n", prop.warpSize);
		printf("  Max threads per block:                %d\n", prop.maxThreadsPerBlock);
		printf("  Max thread dimensions:                (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("  Max grid dimensions:                  (%d, %d, %d)\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	}


	// select device satisfied given conditions
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1 ;
	prop.major = 3 ;
	cudaChooseDevice(&dev, &prop);
	printf("ID of selected CUDA device: %d\n", dev);
	cudaSetDevice(dev);

	return 0;
}
