#!/usr/bin/env python

import scipy as sc
from matplotlib.pyplot import *
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

class Wave2D:
	def __init__(s, nx, ny):
		s.nx, s.ny = nx, ny
		s._mem_allocate()


	def _mem_allocate(s):
		s.c = sc.ones((s.nx,s.ny),'f')*0.25
		s.f = sc.zeros_like(s.c)
		s.c_gpu = cuda.to_device(s.c)
		s.f_gpu = cuda.to_device(s.f)
		s.g_gpu = cuda.to_device(s.f)


	def set_src_pt(s, src_x, src_y):
		s.src_x, s.src_y = src_x, src_y


	def prepare_kernel(s):
		mod = SourceModule("""
			__global__ void update_src(int idx, int tstep, float *f) {
				f[idx] += sin(0.1*tstep);
			}
			__global__ void update(int nx, int ny, float *c, float *f, float *g) {
				int tx = threadIdx.x;
				int idx = blockIdx.x*blockDim.x + tx;

				extern __shared__ float gs[];
				gs[tx+1] = g[idx];

				int i = idx/ny, j = idx%ny;
				if(j>0 && j<ny-1) {
					if(tx==0) gs[tx]=g[idx-1];
					if(tx==blockDim.x-1) gs[tx+2]=g[idx+1];
				}
				__syncthreads();

				if(i>0 && j>0 && i<nx-1 && j<ny-1) {
					f[idx] = c[idx]*(g[idx+ny]+g[idx-ny]+gs[tx+2]+gs[tx]-4*gs[tx+1])+2*gs[tx+1]-f[idx];
				}
			}
			""")
		s.update_src = mod.get_function("update_src")
		s.update = mod.get_function("update")

		Db, s.Dg = (256,1,1), (s.nx*s.ny/256+1,1)
		s.nx, s.ny = sc.int32(s.nx), sc.int32(s.ny)
		s.src_pt = sc.int32(s.src_x*s.ny + s.src_y)

		s.update_src.prepare("iiP",block=(1,1,1))
		s.update.prepare("iiPPP",block=Db,shared=(256+2)*4)

		
	def propagate(s, tn):
		s.update_src.prepared_call((1,1),s.src_pt,sc.int32(tn),s.f_gpu)
		s.update.prepared_call(s.Dg,s.nx,s.ny,s.c_gpu,s.f_gpu,s.g_gpu)
		s.update.prepared_call(s.Dg,s.nx,s.ny,s.c_gpu,s.g_gpu,s.f_gpu)


if __name__ == '__main__':
	nx, ny = 3002, 3002
	x0, y0 = 2001, 1501
	dx = 20				# slit thickness
	w, d = 30, 170		# slit width, distance

	S = Wave2D(nx,ny)

	slx = slice(x0, x0+dx)
	sly1 = slice(y0-(d+w)/2, y0+(d+w)/2)
	sly2 = slice(y0-(d-w)/2, y0+(d-w)/2)
	S.c[slx,:] = 0
	S.c[slx,sly1] = 0.25
	S.c[slx,sly2] = 0
	cuda.memcpy_htod(S.c_gpu,S.c)

	S.set_src_pt(1001, 1501)
	S.prepare_kernel()

	ion()
	from matplotlib.patches import Rectangle
	rect1 = Rectangle((100,0),20,150,facecolor='0.4')
	rect2 = Rectangle((100,180),20,140,facecolor='0.4')
	rect3 = Rectangle((100,350),20,150,facecolor='0.4')
	gca().add_patch(rect1)
	gca().add_patch(rect2)
	gca().add_patch(rect3)
	imsh = imshow(sc.ones((500,500),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.1)
	#imsh = imshow(c.T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.1)
	colorbar()

	for tn in xrange(3000):
		S.propagate(tn)

		if tn>1000:
		#if tstep%10 == 0:
			cuda.memcpy_dtoh(S.f,S.f_gpu)
			imsh.set_array( sc.sqrt(S.f[x0-100:x0+400,y0-250:y0+250].T**2) )
			#imsh.set_array( sc.sqrt(f.T**2) )
			draw()
			#savefig('./png-wave/%.5d.png' % tstep) 
