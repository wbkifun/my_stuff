#define Dx 256

extern "C" {

__global__ void advance_src(int nx, int ny, int tn, double *f) {
    int tid0 = nx/3 + (ny/2)*nx;

	f[tid0] += sin(0.1*tn);
}



__global__ void advance(int nx, int ny, double *c, double *f, double *g) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	int i = tid%nx;
	int j = tid/nx;
	
	
	if( i>0 && i<nx-1 && j>0 && j<ny-1 )
		f[tid] = c[tid]*(g[tid+1] + g[tid-1] + g[tid+nx] + g[tid-nx] - 4*g[tid]) 
            + 2*g[tid] - f[tid];
}



__global__ void advance_smem(int nx, int ny, double *c, double *f, double *g) {
    int tx = threadIdx.x;
	int tid = blockIdx.x*blockDim.x + tx;
	int i = tid%nx;
	int j = tid/nx;
	
    __shared__ double sm[Dx+2];
    double *s = &sm[1];
    s[tx] = g[tid];
    if( tx==0 && i>0 ) s[-1] = g[tid-1];
    if( tx==Dx-1 && i<nx-1 ) s[Dx] = g[tid+1];
    __syncthreads();
	
	if( i>0 && i<nx-1 && j>0 && j<ny-1 )
		f[tid] = c[tid]*(s[tx+1] + s[tx-1] + g[tid+nx] + g[tid-nx] - 4*s[tx]) 
            + 2*s[tx] - f[tid];
}

}
