#include <stdio.h>

int main(int argc, char* argv[]) {
	size_t *free, *total;
	int gpu_id;

	if (argc == 1) {
		fputs("Error: GPU id number is required.\n", stderr);
		exit(1);
	}

	gpu_id = atoi(argv[1]);
	cudaSetDevice(gpu_id);
	cudaMemGetInfo((size_t*)&free, (size_t*)&total);

	printf ("\ttotal\t\tfree\t\tused\n");
	printf ("GPU %d\t%lu\t%lu\t%lu\n", gpu_id, (size_t)total, (size_t)free, (size_t)total-(size_t)free);
}
