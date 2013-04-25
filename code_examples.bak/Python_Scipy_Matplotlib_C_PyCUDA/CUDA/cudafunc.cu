__global__ void initmem( int Ntot, float *a ) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if ( idx < Ntot ) a[idx] = 0;
}


__global__ void diff( int Nx, int Ny, float dx, float *a, float *da ) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if ( idx < (Nx-1)*(Ny-1) ) da[idx] = (1/dx)*( a[idx+Ny+1] - a[idx] );
}
