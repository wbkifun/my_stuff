#include "param1.h"
#include "amb_ext1.h"


extern "C" {


__global__ void amb(int shift_gid, int nx, double *a, double *b, double *c) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x + shift_gid;
	
	if (gid >= nx) return;

	bmc(LLL, b, c);
	c[gid] = KK*a[gid] + c[gid];
}


}	// extern "C"
