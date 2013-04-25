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

#from core_numpy import advance
from core_fortran import advance


# Setup
nx, ny = 2400, 2000
tmax, tgap = 1500, 40
c = numpy.ones((nx,ny), order='F')*0.25
f = numpy.zeros_like(c, order='F')
g = numpy.zeros_like(c, order='F')


# Plot using the matplotlib
plt.ion()
fig = plt.figure(figsize=(8,10))
ax1 = fig.add_subplot(2,1,1)
ax1.plot([nx//2,nx//2], [0,ny], '--k')
imag = ax1.imshow(c.T, origin='lower', vmin=-0.1, vmax=0.1)
fig.colorbar(imag)
ax2 = fig.add_subplot(2,1,2)
line, = ax2.plot(c[nx//2,:])
ax2.set_xlim(0, ny)
ax2.set_ylim(-0.1, 0.1)


# Main loop for the time evolution
t0 = time()
f_avg = numpy.zeros(ny)
for tn in xrange(1,tmax+1):
    #g[nx//3,ny//2] += numpy.sin(0.05*numpy.pi*tn) 
    g[nx//3,ny//2+100] += numpy.sin(0.05*numpy.pi*tn) 
    g[nx//3,ny//2-100] += numpy.sin(0.05*numpy.pi*tn) 
    advance(c, f, g)
    advance(c, g, f)
    f_avg[:] += f[nx//2,:]**2

    if tn%tgap == 0:
        print "%d (%d %%)" % (tn, tn/tmax*100)
        imag.set_array(f.T)
        line.set_ydata(f_avg)
        f_avg[:] = 0
        plt.draw()
        #plt.savefig('./png/%.5d.png' % tn) 

print "throughput: %1.3f Mcell/s" % (nx*ny*tmax/(time()-t0)/1e6)
