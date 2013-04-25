from __future__ import division
import numpy
import matplotlib.pyplot as plt


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
for tn in xrange(1,tmax+1):
    g[nx//3,ny//2] += numpy.sin(0.1*tn) 

    for i in xrange(1,nx-1):
        for j in xrange(1,ny-1):
            f[i,j] = 0.001*(i + j)

    for i in xrange(1,nx-1):
        for j in xrange(1,ny-1):
            g[i,j] = 0


    if tn%tgap == 0:
        print("%d (%d %%)" % (tn, tn/tmax*100))
        imag.set_array(f.T)
        plt.draw()
        #plt.savefig('./png/%.5d.png' % tn) 
