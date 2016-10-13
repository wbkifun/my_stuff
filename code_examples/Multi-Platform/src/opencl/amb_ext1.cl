void bmc(double ll, __global double *b, __global double *c) {
	int gid = get_global_id(0);

	c[gid] = ll*b[gid] + mc(MM, c[gid]);
}
