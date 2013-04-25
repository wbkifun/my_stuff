#!/usr/bin/env python

import scipy as sc
from matplotlib.pyplot import *
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

nx, ny = 3000, 3000
c = sc.ones((nx,ny),'f')*0.25
f = sc.zeros_like(c)

c_gpu = cuda.mem_alloc(c.nbytes)
f_gpu = cuda.mem_alloc(c.nbytes)
g_gpu = cuda.mem_alloc(c.nbytes)

c[nx/3*2:nx/3*2+20,:] = 0
c[nx/3*2:nx/3*2+20,ny/2-100:ny/2+100] = 0.25
c[nx/3*2:nx/3*2+20,ny/2-70:ny/2+70] = 0
cuda.memcpy_htod(c_gpu,c)

mod = SourceModule("""
	__global__ void initzero(int n, float *f, float *g) {
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		if(idx<n) { f[idx] = 0; g[idx] = 0; }
	}
	__global__ void update_src(int idx, int tstep, float *g) {
		g[idx] += sin(0.1*tstep);
	}
	__global__ void update(int nx, int ny, float *c, float *f, float *g) {
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		int i = idx/ny, j = idx%ny;
		if(i>0 && j>0 && i<nx-1 && j<ny-1) {
			f[idx] = c[idx]*(g[idx+ny]+g[idx-ny]+g[idx+1]+g[idx-1]-4*g[idx])+2*g[idx]-f[idx];
		}
	}
	""")
initzero = mod.get_function("initzero")
update_src = mod.get_function("update_src")
update = mod.get_function("update")
"""
ion()
from matplotlib.patches import Rectangle
rect1 = Rectangle((100,0),20,150,facecolor='0.4')
rect2 = Rectangle((100,180),20,140,facecolor='0.4')
rect3 = Rectangle((100,350),20,150,facecolor='0.4')
gca().add_patch(rect1)
gca().add_patch(rect2)
gca().add_patch(rect3)
imsh = imshow(sc.ones((500,500),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.1)
colorbar()
"""

Db, Dg = (256,1,1), (nx*ny/256+1,1)
nx, ny = sc.int32(nx), sc.int32(ny)
src_pt = sc.int32((nx/3)*nx+ny/2)

initzero(nx*ny,f_gpu,g_gpu,block=Db,grid=Dg)
for tstep in xrange(1000):
	update_src(src_pt,sc.int32(tstep),g_gpu,block=(1,1,1),grid=(1,1))
	update(nx,ny,c_gpu,f_gpu,g_gpu,block=Db,grid=Dg)
	update(nx,ny,c_gpu,g_gpu,f_gpu,block=Db,grid=Dg)

	"""
	if tstep>950:
	#if tstep%10 == 0:
		print tstep
		cuda.memcpy_dtoh(f,f_gpu)
		imsh.set_array( sc.sqrt(f[nx/3*2-100:nx/3*2+400,ny/2-250:ny/2+250].T**2) )
		draw()
		savefig('./png-wave/%.5d.png' % tstep) 
	"""
