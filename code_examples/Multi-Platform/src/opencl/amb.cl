__kernel void amb(int nx, __global double *a, __global double *b, __global double *c) {
	int gid = get_global_id(0);
	
	if (gid >= nx) return;

	bmc(LLL, b, c);
	c[gid] = KK*a[gid] + c[gid];
}
