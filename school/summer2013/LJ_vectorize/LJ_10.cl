#pragma OPENCL EXTENSION cl_khr_fp64: enable 


__kernel void force(const int n, const double lje, const double ljs, 
					__global double *x, __global double *y, __global double *z,
					__global double *fx, __global double *fy, __global double *fz) {
	int gid = get_global_id(0);
	int j;
	double dx, dy, dz, tx, ty, tz;
	double r, r2, lr, lr6, lr12, pe;

	if (gid<n) {
		tx = 0;
		ty = 0;
		tz = 0;

		for (j=0; j<n; j++) {
			if (j==gid) continue;

			dx = x[j] - x[gid];
			dy = y[j] - y[gid];
			dz = z[j] - z[gid];
			r = sqrt(dx*dx + dy*dy + dz*dz);
			r2 = r*r;
			
			lr = ljs/r;
			lr6 = lr * lr * lr * lr * lr * lr;
			lr12 = lr6 * lr6;
			pe = 2*lr12 - lr6;

			tx += pe*dx/r2;
			ty += pe*dy/r2;
			tz += pe*dz/r2;
		}

		fx[gid] = -24 * lje * tx;
		fy[gid] = -24 * lje * ty;
		fz[gid] = -24 * lje * tz;
	}
}




__kernel void solve(const int n, const int dt, const double em, 
					__global double *x, __global double *y, __global double *z,
					__global double *vx, __global double *vy, __global double *vz,
					__global double *fx, __global double *fy, __global double *fz) {
	int gid = get_global_id(0);

	if (gid<n) {
		x[gid] += vx[gid]*dt;
		y[gid] += vy[gid]*dt;
		z[gid] += vz[gid]*dt;

		vx[gid] += fx[gid]*dt/em;
		vy[gid] += fy[gid]*dt/em;
		vz[gid] += fz[gid]*dt/em;
	}
}



__kernel void energy(const int n, const double em, 
			        const double lje, const double ljs, 
					__global double *x, __global double *y, __global double *z,
					__global double *vx, __global double *vy, __global double *vz,
					__global double *ke_group, __global double *pe_group) {
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int grp_id = get_group_id(0);
	int j;
	double dx, dy, dz;
	double r, lr, lr6, lr12;
	__local double ke, pe;


	if (gid<n) {
		ke = 0;
		pe = 0;

		// kinetic energy
		ke +=  0.5 * em * (vx[gid]*vx[gid] + vy[gid]*vy[gid] + vz[gid]*vz[gid]);


		// potential energy
		for (j=0; j<n; j++) {
			if (j==gid) continue;

			dx = x[j] - x[gid];
			dy = y[j] - y[gid];
			dz = z[j] - z[gid];
			r = sqrt(dx*dx + dy*dy + dz*dz);

			lr = ljs/r;
			lr6 = lr * lr * lr * lr * lr * lr;
			lr12 = lr6 * lr6;
			pe += 0.5 * (4*lje*lr12 - lr6);
		}


		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid==0) {
			ke_group[grp_id] = ke;
			pe_group[grp_id] = pe;
		}
	}
}
