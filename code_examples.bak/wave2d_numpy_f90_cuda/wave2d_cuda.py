#------------------------------------------------------------------------------
# Author : Ki-Hwan Kim (kh.kim@kiaps.org)
#
# Written date : 2010. 6. 17
# Modify date  : 2012. 9. 17
#
# Copyright    : GNU GPL
#
# Description  :
# Solve the 2-D wave equation with the FD(Finite-Difference) scheme
#
# These are educational codes to study the scientific python programming. 
# Step 1: Using the numpy
# Step 2: Convert the hotspot to the Fortran code using F2PY
# Step 3: Convert the hotspot to the CUDA code using PyCUDA
#------------------------------------------------------------------------------

from __future__ import division
import numpy
import matplotlib.pyplot as plt
from time import time
import pycuda.driver as cuda
import pycuda.autoinit


# Setup
nx, ny = 1200, 1000
tmax, tgap = 500, 100
c = numpy.ones((nx,ny), order='F')*0.25
f = numpy.zeros_like(c, order='F')
c_gpu = cuda.to_device(c)
f_gpu = cuda.to_device(f)
g_gpu = cuda.to_device(f)

mod = cuda.module_from_file('core.cubin')
advance_src = mod.get_function('advance_src')
#advance = mod.get_function('advance')
advance = mod.get_function('advance_smem')


# Plot using the matplotlib
plt.ion()
imag = plt.imshow(c.T, origin='lower', vmin=-0.2, vmax=0.2)
plt.colorbar()


# Main loop for the time evolution
inx, iny = numpy.int32(nx), numpy.int32(ny)
bs, gs = (256,1,1), (nx*ny//256+1,1)
t0 = time()
for tn in xrange(1,tmax+1):
    advance_src(inx, iny, numpy.int32(tn), g_gpu, block=(1,1,1), grid=(1,1))
    advance(inx, iny, c_gpu, f_gpu, g_gpu, block=bs, grid=gs)
    advance(inx, iny, c_gpu, g_gpu, f_gpu, block=bs, grid=gs)

    '''
    if tn%tgap == 0:
        cuda.memcpy_dtoh(f,f_gpu)

        print "%d (%d %%)" % (tn, tn/tmax*100)
        imag.set_array(f.T)
        plt.draw()
        #plt.savefig('./png/%.5d.png' % tn) 
    '''

cuda.memcpy_dtoh(f,f_gpu)
print "throughput: %1.3f Mcell/s" % (nx*ny*tmax/(time()-t0)/1e6)
