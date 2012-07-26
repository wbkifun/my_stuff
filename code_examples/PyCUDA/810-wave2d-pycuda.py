#!/usr/bin/env python

import numpy as np
import sys
from matplotlib.pyplot import *
from datetime import datetime
import pycuda.driver as cuda
import pycuda.autoinit


kernels ="""
__global__ void update(int nx, int ny, float *c, float *f, float *g) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i = idx/ny;

	if( i>0 && i<nx-1 )
		f[idx] = c[idx]*(g[idx+ny] + g[idx-ny] + g[idx+1] + g[idx-1] - 4*g[idx]) + 2*g[idx] - f[idx];
}

__global__ void update_src(int nx, int ny, float tn, float *f) {
	f[400*ny+500] += sin(0.1*tn);
}

"""
from pycuda.compiler import SourceModule
mod = SourceModule(kernels)
update = mod.get_function("update")
update_src = mod.get_function("update_src")


nx, ny = 1000, 1000
tmax = 200
c = np.ones((nx,ny),'f')*0.25
f = np.zeros_like(c)
c_gpu = cuda.to_device(c)
f_gpu = cuda.to_device(f)
g_gpu = cuda.to_device(f)

ion()
imsh = imshow(np.ones_like(c).T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.2)
colorbar()

t1 = datetime.now()
for tn in range(1,tmax+1):
	update_src(np.int32(nx), np.int32(ny), np.float32(tn), g_gpu, block=(1,1,1), grid=(1,1))
	update(np.int32(nx), np.int32(ny), c_gpu, f_gpu, g_gpu, block=(256,1,1), grid=(nx*ny/256+1,1))
	update(np.int32(nx), np.int32(ny), c_gpu, g_gpu, f_gpu, block=(256,1,1), grid=(nx*ny/256+1,1))

	if tn%10 == 0:
		print "tstep =\t%d/%d (%d %%)\r" % (tn, tmax, float(tn)/tmax*100),
		sys.stdout.flush()
		cuda.memcpy_dtoh(f,f_gpu)
		imsh.set_array( np.sqrt(f.T**2) )
		draw()
		#savefig('./png/%.5d.png' % tn) 

print ''
print datetime.now() - t1
