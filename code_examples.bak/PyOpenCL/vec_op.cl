__kernel void vecadd_gpu(
			__global float* a,
			__global const float* b) {
	const int idx = get_global_id(0);

	if(idx < NX) {
		a[idx] += b[idx];
	}
}


__kernel void vecsub_gpu(
			__global float* a,
			__global const float* b) {
	const int idx = get_global_id(0);

	if(idx < NX) {
		a[idx] -= b[idx];
	}
}
