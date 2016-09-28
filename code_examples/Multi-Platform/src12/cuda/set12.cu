extern "C" {


__global__ void calc_divv(int shift_gid, int np, int nlev, int nelem, double *ru) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x + shift_gid;
	
	if (idx >= np*np*(nlev+1)*nelem) return;

	ru[idx] = 1.2;
}


}	// extern "C"
