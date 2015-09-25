void daxpy(int n, double *ret, double a, double *x, double *y) {
    int i;

    for(i=0; i<n; i++)
        ret[i] = a*x[i] + y[i];
}




void rk4sum(int n, double dt, double *k1, double *k2, double *k3, double *k4, double *ret) {
    int i;

    for(i=0; i<n; i++) 
        ret[i] += (dt/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
}
