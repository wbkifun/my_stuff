__global__ void daxpy(int n, double *ret, double a, double *x, double *y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) return;
    ret[i] = a*x[i] + y[i];
}




__global__ void rk4sum(int n, double dt, double *k1, double *k2, double *k3, double *k4, double *ret) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) return;
    ret[i] += (dt/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
}
