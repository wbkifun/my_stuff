/*--------------------------------------------------------------------------
# File Name : wave2d_core.cl
#
# Author : Ki-Hwan Kim (wbkifun@nate.com)
# 
# Written date :	2011. 6. 23
# Modify date :		
#
# Copyright : GNU GPL
--------------------------------------------------------------------------*/

__kernel void update(int nx, int ny, __global float *c, __global float *f, __global float *g) {
	int gid = get_global_id(0);
	int i = gid/ny;
	int j = gid%ny;
	
	if( i>0 && i<nx-1 && j>0 && j<ny-1 )
		f[gid] = c[gid]*(g[gid+ny] + g[gid-ny] + g[gid+1] + g[gid-1] - 4*g[gid]) + 2*g[gid] - f[gid];
}


__kernel void update_src(int nx, int ny, float tn, __global float *f) {
	f[300*ny+400] += sin(0.1*tn);
}
