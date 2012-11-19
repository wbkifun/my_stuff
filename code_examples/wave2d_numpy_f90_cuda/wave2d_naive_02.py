from __future__ import division
import numpy
import matplotlib.pyplot as plt
from time import time



def advance(c, f, g):
    for i in xrange(1,nx-1):
        for j in xrange(1,ny-1):
            f[i,j] = c[i,j]*(g[i+1,j] + g[i-1,j] + g[i,j+1] + g[i,j-1] - 4*g[i,j]) \
                    + 2*g[i,j] - f[i,j]


# Setup
nx, ny = 240, 200
tmax, tgap = 100, 10
c = numpy.ones((nx,ny))*0.25
f = numpy.zeros_like(c)
g = numpy.zeros_like(c)


# Plot using the matplotlib
plt.ion()
imag = plt.imshow(c.T, origin='lower', vmin=-0.2, vmax=0.2)
plt.colorbar()


# Main loop for the time evolution
t0 = time()
for tn in xrange(1,tmax+1):
    g[nx//3,ny//2] += numpy.sin(0.1*tn) 
    advance(c, f, g)
    advance(c, g, f)

    if tn%tgap == 0:
        print("%d (%d %%)" % (tn, tn/tmax*100))
        imag.set_array(f.T)
        plt.draw()
        #plt.savefig('./png/%.5d.png' % tn) 

print("throughput: %1.3f Mcell/s" % (nx*ny*tmax/(time()-t0)/1e6))
