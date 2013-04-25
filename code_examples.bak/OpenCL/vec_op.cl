__kernel void vecadd_gpu(
		const int nx,
		__global float* a,
		__global const float* b) {
	const int idx = get_global_id(0);

	if(idx < nx) {
		a[idx] += b[idx];
	}
}


__kernel void vecsub_gpu(
		const int nx,
		__global float* a,
		__global const float* b) {
	const int idx = get_global_id(0);

	if(idx < nx) {
		a[idx] -= b[idx];
	}
}
