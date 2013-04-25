/*--------------------------------------------------------------------------
# File Name : vecadd.cl
#
# Author : Ki-Hwan Kim (wbkifun@nate.com)
# 
# Written date :	2011. 6. 22
# Modify date :		2011. 8. 9	(Add while statement)
#
# Copyright : GNU GPL
--------------------------------------------------------------------------*/

__kernel void vecadd(int nmax, __global const float *a, __global const float *b, __global float *c) {
	int gid = get_global_id(0);
	
	while ( gid < nmax ) {
		c[gid] = a[gid] + b[gid];
		gid += get_global_size(0);
	}
}
