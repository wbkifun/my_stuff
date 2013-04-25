__kernel void vecadd_gpu(
		const int nx,
		__global const float* a,
		__global const float* b,
		__global float* c) {
	const int idx = get_global_id(0);

	if(idx < nx) {
		c[idx] = a[idx] + b[idx];
	}
}
