#define Dx 256

extern "C" {

__global__ void advance_src(int nx, int ny, int tn, double *f) {
    int idx0 = nx/3 + (ny/2)*nx;

	f[idx0] += sin(0.1*tn);
}



__global__ void advance(int nx, int ny, double *c, double *f, double *g) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = idx%nx;
	int j = idx/nx;
	
	
	if( i>0 && i<nx-1 && j>0 && j<ny-1 )
		f[idx] = c[idx]*(g[idx+1] + g[idx-1] + g[idx+nx] + g[idx-nx] - 4*g[idx]) 
            + 2*g[idx] - f[idx];
}



__global__ void advance_smem(int nx, int ny, double *c, double *f, double *g) {
    int tx = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + tx;
	int i = idx%nx;
	int j = idx/nx;
	
    __shared__ double sm[Dx+2];
    double *s = &sm[1];
    s[tx] = g[idx];
    if( tx==0 && i>0 ) s[-1] = g[idx-1];
    if( tx==Dx-1 && i<nx-1 ) s[Dx] = g[idx+1];
    __syncthreads();
	
	if( i>0 && i<nx-1 && j>0 && j<ny-1 )
		f[idx] = c[idx]*(s[tx+1] + s[tx-1] + g[idx+nx] + g[idx-nx] - 4*s[tx]) 
            + 2*s[tx] - f[idx];
}

}
