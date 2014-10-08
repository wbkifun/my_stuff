#------------------------------------------------------------------------------
# Author : Ki-Hwan Kim (kh.kim@kiaps.org)
#
# Written date : 2010. 6. 17
# Modify date  : 2012. 9. 25 (use C array)
#
# Copyright    : GNU GPL
#
# Description  :
# Solve the 2-D wave equation with the FD(Finite-Difference) scheme
#
# These are educational codes to study the scientific python programming. 
# Step 1: Using the numpy
# Step 2: Convert the hotspot to the Fortran code using F2PY
#------------------------------------------------------------------------------

from __future__ import division
import numpy
import matplotlib.pyplot as plt
from time import time

from core_fortran import update


# Setup
nx, ny = 1200, 1000
tmax, tgap = 500, 10
c = numpy.ones((nx,ny), order='F')*0.25
f = numpy.zeros_like(c, order='F')
g = numpy.zeros_like(c, order='F')


# Plot using the matplotlib
plt.ion()
imag = plt.imshow(c.T, origin='lower', vmin=-0.2, vmax=0.2)
plt.colorbar()


# Main loop for the time evolution
t0 = time()
for tn in xrange(1,tmax+1):
    g[nx//3,ny//2] += numpy.sin(0.1*tn) 
    update(c, f, g)
    update(c, g, f)

    if tn%tgap == 0:
        print "%d (%d %%)" % (tn, tn/tmax*100)
        imag.set_array(f.T)
        plt.draw()
        #plt.savefig('./png/%.5d.png' % tn) 

print "throughput: %1.3f Mcell/s" % (nx*ny*tmax/(time()-t0)/1e6)
plt.show(True)
