#!/usr/bin/env python

import scipy as sc
from matplotlib.pyplot import *
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

class Wave2D:
	def __init__(s, nx, ny):
		s.nx, s.ny = nx, ny


	def mem_allocate(s):
		s.c = sc.ones((s.nx,s.ny),'f')*0.25
		s.f = sc.zeros_like(s.c)
		s.c_gpu = cuda.mem_alloc(s.c.nbytes)
		s.f_gpu = cuda.mem_alloc(s.c.nbytes)
		s.g_gpu = cuda.mem_alloc(s.c.nbytes)


	def set_geometry(s):
		s.c[s.nx/3*2:s.nx/3*2+20,:] = 0
		s.c[s.nx/3*2:s.nx/3*2+20,s.ny/2-100:s.ny/2+100] = 0.25
		s.c[s.nx/3*2:s.nx/3*2+20,s.ny/2-70:s.ny/2+70] = 0
		cuda.memcpy_htod(s.c_gpu,s.c)


	def prepare_kernel(s):
		mod = SourceModule("""
			__global__ void initzero(int n, float *f, float *g) {
				int idx = blockIdx.x*blockDim.x + threadIdx.x;
				if(idx<n) { f[idx] = 0; g[idx] = 0; }
			}
			__global__ void update_src(int idx, int tstep, float *g) {
				g[idx] += sin(0.1*tstep);
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
		s.initzero = mod.get_function("initzero")
		s.update_src = mod.get_function("update_src")
		s.update = mod.get_function("update")

		Db, s.Dg = (256,1,1), (s.nx*s.ny/256+1,1)
		s.nx, s.ny = sc.int32(s.nx), sc.int32(s.ny)
		s.src_pt = sc.int32((s.nx/3)*s.nx+s.ny/2)

		s.initzero(s.nx*s.ny,s.f_gpu,s.g_gpu,block=Db,grid=s.Dg)
		s.update_src.prepare("iiP",block=(1,1,1))
		s.update.prepare("iiPPP",block=Db,shared=(256+2)*4)

		
	def propagate(s, tn):
		s.update_src.prepared_call((1,1),s.src_pt,sc.int32(tn),s.g_gpu)
		s.update.prepared_call(S.Dg,s.nx,s.ny,s.c_gpu,s.f_gpu,s.g_gpu)
		s.update.prepared_call(S.Dg,s.nx,s.ny,s.c_gpu,s.g_gpu,s.f_gpu)


S = Wave2D(3000,3000)
S.mem_allocate()
S.set_geometry()
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
		imsh.set_array( sc.sqrt(S.f[S.nx/3*2-100:S.nx/3*2+400,S.ny/2-250:S.ny/2+250].T**2) )
		#imsh.set_array( sc.sqrt(f.T**2) )
		draw()
		#savefig('./png-wave/%.5d.png' % tstep) 
