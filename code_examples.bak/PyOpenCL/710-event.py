#!/usr/bin/env python

import numpy as np
import sys
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
MAX_GRID = 65535

kernels =""" 
__kernel void update_h( 
		__global float *ex, __global float *ey, __global float *ez,
		__global float *hx, __global float *hy, __global float *hz,
		__global float *chx, __global float *chy, __global float *chz) {
	int idx = get_global_id(0);
	int k = idx/(NXY);
	int j = (idx - k*NXY)/NX;
	int i = idx%NX;

	if( j>0 && k>0 ) hx[idx] -= chx[idx]*( ez[idx] - ez[idx-NX] - ey[idx] + ey[idx-NXY] );
	if( i>0 && k>0 ) hy[idx] -= chy[idx]*( ex[idx] - ex[idx-NXY] - ez[idx] + ez[idx-1] );
	if( i>0 && j>0 ) hz[idx] -= chz[idx]*( ey[idx] - ey[idx-1] - ex[idx] + ex[idx-NX] );
}

__kernel void update_e(
		__global float *ex, __global float *ey, __global float *ez,
		__global float *hx, __global float *hy, __global float *hz,
		__global float *cex, __global float *cey, __global float *cez) {
	int idx = get_global_id(0);
	int k = idx/(NXY);
	int j = (idx - k*NXY)/NX;
	int i = idx%NX;

	if( j<NY-1 && k<NZ-1 ) ex[idx] += cex[idx]*( hz[idx+NX] - hz[idx] - hy[idx+NXY] + hy[idx] );
	if( i<NX-1 && k<NZ-1 ) ey[idx] += cey[idx]*( hx[idx+NXY] - hx[idx] - hz[idx+1] + hz[idx] );
	if( i<NX-1 && j<NY-1 ) ez[idx] += cez[idx]*( hy[idx+1] - hy[idx] - hx[idx+NX] + hx[idx] );
}

__kernel void update_src(float tn, __global float *f) {
	int idx = get_global_id(0);
	int ijk = (NZ/2)*NXY + (NY/2)*NX + idx;

	if( idx < NX ) f[ijk] += sin(0.1*tn);
}
"""


if __name__ == '__main__':
	nx, ny, nz = 256, 256, 240
	nnx, nny, nnz = np.int32(nx), np.int32(ny), np.int32(nz)

	print 'dim (%d, %d, %d)' % (nx, ny, nz)
	total_bytes = nx*ny*nz*4*12
	if total_bytes/(1024**3) == 0:
		print 'mem %d MB' % ( total_bytes/(1024**2) )
	else:
		print 'mem %1.2f GB' % ( float(total_bytes)/(1024**3) )

	# memory allocate
	f = np.zeros((nx,ny,nz), 'f', order='F')
	cf = np.ones_like(f)*0.5

	mf = cl.mem_flags
	ex_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	ey_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	ez_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	hx_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	hy_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)
	hz_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f)

	cex_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	cey_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	cez_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	chx_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	chy_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)
	chz_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cf)

	# prepare kernels
	kern = kernels.replace('NXY',str(nx*ny)).replace('NX',str(nx)).replace('NY',str(ny)).replace('NZ',str(nz))
	prg = cl.Program(ctx, kern).build()
	update_h = prg.update_h
	update_e = prg.update_e
	update_src = prg.update_src

	Gs = (nx*ny*nz,)	# global size
	Ls = (256,)			# local size

	# prepare for plot
	from matplotlib.pyplot import *
	ion()
	imsh = imshow(np.ones((ny,nz),'f'), cmap=cm.hot, origin='lower', vmin=0, vmax=0.005)
	colorbar()

	# measure kernel execution time
	from datetime import datetime
	t1 = datetime.now()

	# main loop
	tmax = 1000
	evt_src = update_src(queue, (nx,), (256,), np.float32(0), ex_gpu)
	for tn in xrange(1, tmax+1):
		#enqueue_nd_range_kernel(queue, update_h, Gs, Ls, wait_for=None)
		evt_h = update_h(
				queue, Gs, Ls,
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				chx_gpu, chy_gpu, chz_gpu, wait_for=[evt_src])

		evt_e = update_e(
				queue, Gs, Ls,
				ex_gpu, ey_gpu, ez_gpu, hx_gpu, hy_gpu, hz_gpu, 
				cex_gpu, cey_gpu, cez_gpu, wait_for=[evt_h])

		evt_src = update_src(queue, (nx,), (256,), 
				np.float32(tn), ex_gpu, wait_for=[evt_e])

		if tn%10 == 0:
			print "[%d/%d (%d %%)]\r" % (tn, tmax, float(tn)/tmax*100),
			#sys.stdout.flush()
			#cl.enqueue_read_buffer(queue, ex_gpu, f)
			#imsh.set_array( f[nx/2,:,:]**2 )
			#draw()
			#savefig('./png-wave/%.5d.png' % tstep) 

	#cl.enqueue_marker(queue).wait()
	#cl.enqueue_barrier(queue)
	print '\n', datetime.now() - t1
