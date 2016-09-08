#include "param1.h"
#include "apb_ext.h"


extern "C" {


__global__ void apb(int nx, double *a, double *b, double *c) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid >= nx) return;

	c[gid] = KK*a[gid] + bpc(b[gid], c[gid]);
}


}	// extern "C"
