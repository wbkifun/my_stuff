__kernel void calc_divv(int np, int nlev, int nelem, __global double *ru) {
	int idx = get_global_id(0);
	
	if (idx >= np*np*(nlev+1)*nelem) return;

	ru[idx] = 1.2;
}
