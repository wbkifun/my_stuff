__kernel void daxpy(int n, __global double *ret, double a, __global double *x, __global double *y) {
    int i = get_global_id(0);

    if (i >= n) return;
    ret[i] = a*x[i] + y[i];
}




__kernel void rk4sum(int n, double dt, __global double *k1, __global double *k2, __global double *k3, __global double *k4, __global double *ret) {
    int i = get_global_id(0);

    if (i >= n) return;
    ret[i] += (dt/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
}
