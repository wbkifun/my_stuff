from __future__ import division
import numpy
import matplotlib.pyplot as plt



def mandelbrot(x0, x1, y0, y1, res=400, maxiter=1000):
    '''Returns an image of the Mandelbrot fractal'''
    
    x, y = numpy.ogrid[x0:x1:res*1j, y0:y1:res*1j]
    c = x + y*1j
    z = 0
    divtime = numpy.ones(c.shape, dtype=int) * maxiter

    for i in xrange(maxiter):
        z = z**2 + c
        diverge = numpy.abs(z) > 2                 # who is diverging
        z[diverge] = 2                             # avoid diverging too much
        divtime[diverge & (divtime==maxiter)] = i  # note who is diverging now

    return divtime



x0, y0 = (-0.743643887037158704752191506114774, 0.131825904205311970493132056385139)
dx, dy = 0.0001, 0.0001
x1, x2, y1, y2 = x0-dx, x0+dx, y0-dy, y0+dy
plt.imshow(mandelbrot(x1, x2, y1, y2).T, cmap=plt.cm.jet, origin='lower')#, extent=(x1, x2, y1, y2))
plt.colorbar()
plt.show()
