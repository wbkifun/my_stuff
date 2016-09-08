#include "param2.h"
#include "amb_ext2.h"


__device__ void bmc(double ll, double *b, double *c) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	c[gid] = ll*b[gid] + mc(MM, c[gid]);
}
