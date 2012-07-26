#!/usr/bin/env python
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


# setup
nx, ny = 1000, 800
tmax, tgap = 600, 100

# allocation
c0 = 0.25    # ( velocity * (dt/dx) )**2
c = np.ones((nx, ny)) * c0
f = np.zeros_like(c)

c_gpu = cuda.to_device(c)
f_gpu = cuda.to_device(f)
g_gpu = cuda.to_device(f)

# cuda kernels
from pycuda.compiler import SourceModule
kernels = open('pycuda_wave_fd.cu').read()
mod = SourceModule(kernels)
update_core = mod.get_function('update_core')
update_src = mod.get_function('update_src')

# plot
import matplotlib.pyplot as plt
plt.ion()
imag = plt.imshow(f, vmin=-0.1, vmax=0.1)
plt.colorbar()

# main time loop
for tstep in xrange(1, tmax+1):
    update_core(f_gpu, g_gpu, c_gpu, np.int32(nx), np.int32(ny), block=(256,1,1), grid=(nx*ny/256+1,1))
    update_core(g_gpu, f_gpu, c_gpu, np.int32(nx), np.int32(ny), block=(256,1,1), grid=(nx*ny/256+1,1))
    src_val = np.sin(0.4 * tstep)
    src_idx = np.int32( (nx/2)*ny + ny/2 )
    update_src(g_gpu, src_val, src_idx, block=(256,1,1), grid=(1,1))

    if tstep % 100 == 0:
        print('tstep= %d' % tstep)
        cuda.memcpy_dtoh(f, f_gpu)
        imag.set_array(f)
        #plt.savefig('./png/%.4d.png' % tstep)
        plt.draw()
