from __future__ import division
import numpy
import matplotlib.pyplot as plt
from time import time



si = slice(1,-1)
def advance(c, f, g):
    f[si,si] = c[si,si]*(g[2:,si] + g[:-2,si] + g[si,2:] + g[si,:-2] - 4*g[si,si]) \
            + 2*g[si,si] - f[si,si]



# Setup
nx, ny = 480, 400
tmax, tgap = 200, 10
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
