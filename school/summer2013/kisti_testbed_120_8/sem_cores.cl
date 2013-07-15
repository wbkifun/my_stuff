#pragma OPENCL EXTENSION cl_khr_fp64: enable 


__kernel void interact_inner2(const int n, const int ngll, __global const int *mvp_inner2, __global double *var) {
	int gid = get_global_id(0);
	int gi1, gi2;
	int gj1, gj2;
	int elem1, elem2;
	double sv;


	if (gid<n) {
		// dimension (2,2,n)
		gi1   = mvp_inner2[0 + 0*2 + gid*4] - 1;
		gj1   = mvp_inner2[1 + 0*2 + gid*4] - 1;
		elem1 = mvp_inner2[2 + 0*2 + gid*4] - 1;

		gi2   = mvp_inner2[0 + 1*2 + gid*4] - 1;
		gj2   = mvp_inner2[1 + 1*2 + gid*4] - 1;
		elem2 = mvp_inner2[2 + 1*2 + gid*4] - 1;

		sv = (var[gi1 + gj1*ngll + elem1*ngll*ngll]
			+ var[gi2 + gj2*ngll + elem2*ngll*ngll]) / 2;
		var[gi1 + gj1*ngll + elem1*ngll*ngll] = sv;
		var[gi2 + gj2*ngll + elem2*ngll*ngll] = sv;
	}
}




__kernel void interact_inner3(const int n, const int ngll, __global const int *mvp_inner3, __global double *var) {
	int gid = get_global_id(0);
	int gi1, gi2, gi3;
	int gj1, gj2, gj3;
	int elem1, elem2, elem3;
	double sv;


	if (gid<n) {
		// dimension (2,3,n)
		gi1   = mvp_inner3[0 + 0*2 + gid*6] - 1;
		gj1   = mvp_inner3[1 + 0*2 + gid*6] - 1;
		elem1 = mvp_inner3[2 + 0*2 + gid*6] - 1;

		gi2   = mvp_inner3[0 + 1*2 + gid*6] - 1;
		gj2   = mvp_inner3[1 + 1*2 + gid*6] - 1;
		elem2 = mvp_inner3[2 + 1*2 + gid*6] - 1;

		gi3   = mvp_inner3[0 + 2*2 + gid*6] - 1;
		gj3   = mvp_inner3[1 + 2*2 + gid*6] - 1;
		elem3 = mvp_inner3[2 + 2*2 + gid*6] - 1;

		sv = (var[gi1 + gj1*ngll + elem1*ngll*ngll] +
			  var[gi2 + gj2*ngll + elem2*ngll*ngll] +
			  var[gi3 + gj3*ngll + elem3*ngll*ngll]) / 3;
		var[gi1 + gj1*ngll + elem1*ngll*ngll] = sv;
		var[gi2 + gj2*ngll + elem2*ngll*ngll] = sv;
		var[gi3 + gj3*ngll + elem3*ngll*ngll] = sv;
	}
}



__kernel void interact_inner4(const int n, const int ngll, __global const int *mvp_inner4, __global double *var) {
	int gid = get_global_id(0);
	int gi1, gi2, gi3, gi4;
	int gj1, gj2, gj3, gj4;
	int elem1, elem2, elem3, elem4;
	double sv;


	if (gid<n) {
		// dimension (2,4,n)
		gi1   = mvp_inner4[0 + 0*2 + gid*8] - 1;
		gj1   = mvp_inner4[1 + 0*2 + gid*8] - 1;
		elem1 = mvp_inner4[2 + 0*2 + gid*8] - 1;

		gi2   = mvp_inner4[0 + 1*2 + gid*8] - 1;
		gj2   = mvp_inner4[1 + 1*2 + gid*8] - 1;
		elem2 = mvp_inner4[2 + 1*2 + gid*8] - 1;

		gi3   = mvp_inner4[0 + 2*2 + gid*8] - 1;
		gj3   = mvp_inner4[1 + 2*2 + gid*8] - 1;
		elem3 = mvp_inner4[2 + 2*2 + gid*8] - 1;

		gi4   = mvp_inner4[0 + 3*2 + gid*8] - 1;
		gj4   = mvp_inner4[1 + 3*2 + gid*8] - 1;
		elem4 = mvp_inner4[2 + 3*2 + gid*8] - 1;

		sv = (var[gi1 + gj1*ngll + elem1*ngll*ngll] +
			  var[gi2 + gj2*ngll + elem2*ngll*ngll] +
			  var[gi3 + gj3*ngll + elem3*ngll*ngll] +
			  var[gi4 + gj4*ngll + elem4*ngll*ngll]) / 4;
		var[gi1 + gj1*ngll + elem1*ngll*ngll] = sv;
		var[gi2 + gj2*ngll + elem2*ngll*ngll] = sv;
		var[gi3 + gj3*ngll + elem3*ngll*ngll] = sv;
		var[gi4 + gj4*ngll + elem4*ngll*ngll] = sv;
	}
}



__kernel void compute_rhs(const int nelem, const int ngll, 
						  __global const double *dvvt, 
						  __global const double *jac, 
						  __global const double *Ainv, 
						  __global const double *vel, 
						  __global const double *psi, 
						  __global double *ret_psi) {
	int gid = get_global_id(0);
	int n = ngll*ngll*nelem;
	int gi, gj, ie;
	int k;
	
	__local double vpsix[64], vpsiy[64];
	//double vpsix[64], vpsiy[64];
	double tmpx, tmpy;
	//double tmpx0, tmpx1, tmpx2, tmpx3, tmpx4, tmpx5, tmpx6, tmpx7;
	//double tmpy0, tmpy1, tmpy2, tmpy3, tmpy4, tmpy5, tmpy6, tmpy7;


	ie = gid/64;
	gj = (gid - ie*64)/8;
	gi = gid%8;

  	if (gid<n) {
		vpsix[gi + gj*8] = ( Ainv[0 + 0*2 + gi*4 + gj*32 + ie*256] *
                              vel[0 + gi*2 + gj*16 + ie*128] +
                             Ainv[0 + 1*2 + gi*4 + gj*32 + ie*256] *
                              vel[1 + gi*2 + gj*16 + ie*128] ) * psi[gid];

		vpsiy[gi + gj*8] = ( Ainv[1 + 0*2 + gi*4 + gj*32 + ie*256] *
                              vel[0 + gi*2 + gj*16 + ie*128] +
                             Ainv[1 + 1*2 + gi*4 + gj*32 + ie*256] *
                              vel[1 + gi*2 + gj*16 + ie*128] ) * psi[gid];

		tmpx = 0;
		tmpy = 0;
		for (k=0; k<8; k++) {
        	tmpx += dvvt[k + gi*8] * vpsix[k + gj*8];
        	tmpy += vpsiy[gi + k*8] * dvvt[k + gj*8];
	    }
		/*
		tmpx0 = dvvt[0 + gi*8] * vpsix[0 + gj*8];
		tmpx1 = dvvt[1 + gi*8] * vpsix[1 + gj*8];
		tmpx2 = dvvt[2 + gi*8] * vpsix[2 + gj*8];
		tmpx3 = dvvt[3 + gi*8] * vpsix[3 + gj*8];
		tmpx4 = dvvt[4 + gi*8] * vpsix[4 + gj*8];
		tmpx5 = dvvt[5 + gi*8] * vpsix[5 + gj*8];
		tmpx6 = dvvt[6 + gi*8] * vpsix[6 + gj*8];
		tmpx7 = dvvt[7 + gi*8] * vpsix[7 + gj*8];

		tmpy0 = vpsiy[gi + 0*8] * dvvt[0 + gj*8];
		tmpy1 = vpsiy[gi + 1*8] * dvvt[1 + gj*8];
		tmpy2 = vpsiy[gi + 2*8] * dvvt[2 + gj*8];
		tmpy3 = vpsiy[gi + 3*8] * dvvt[3 + gj*8];
		tmpy4 = vpsiy[gi + 4*8] * dvvt[4 + gj*8];
		tmpy5 = vpsiy[gi + 5*8] * dvvt[5 + gj*8];
		tmpy6 = vpsiy[gi + 6*8] * dvvt[6 + gj*8];
		tmpy7 = vpsiy[gi + 7*8] * dvvt[7 + gj*8];

		tmpx = tmpx0 + tmpx1 + tmpx2 + tmpx3 + tmpx4 + tmpx5 + tmpx6 + tmpx7;
		tmpy = tmpy0 + tmpy1 + tmpy2 + tmpy3 + tmpy4 + tmpy5 + tmpy6 + tmpy7;
		*/
		ret_psi[gid] = -(tmpx + tmpy) / jac[gid];
	}
}



__kernel void add(const int n, const double coeff, __global const double *k, __global const double *psi, __global double *ret_psi) {
	int gid = get_global_id(0);

	if (gid<n) {
		ret_psi[gid] = coeff * k[gid] + psi[gid];
	}
}



__kernel void rk4_add(const int n, double dt, __global const double *k1, __global const double *k2, __global const double *k3, __global const double *k4, __global double *psi) {
	int gid = get_global_id(0);
  	double coeff = dt/6;

	if (gid<n) {
        psi[gid] += coeff * (k1[gid] + 2*k2[gid] + 2*k3[gid] + k4[gid]);
	}
}
