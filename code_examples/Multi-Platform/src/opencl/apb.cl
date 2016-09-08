#include "param1.h"
#include "apb_ext.h"


__kernel void apb(int nx, __global double *a, __global double *b, __global double *c) {
	int gid = get_global_id(0);

	if (gid >= nx) return;

	c[gid] = KK*a[gid] + bpc(b[gid], c[gid]);
}
