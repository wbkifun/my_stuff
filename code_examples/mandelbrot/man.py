from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(x0, x1, y0, y1, res=400, maxiter=1000):
    '''Returns an image of the Mandelbrot fractal'''
    
    x, y = np.ogrid[x0:x1:res*1j, y0:y1:res*1j]
    c = x + y*1j
    z = 0
    divtime = np.ones(c.shape, dtype=int) * maxiter

    for i in xrange(maxiter):
        z = z**2 + c
        diverge = np.abs(z) > 2                    # who is diverging
        z[diverge] = 2                             # avoid diverging too much
        divtime[diverge & (divtime==maxiter)] = i  # note who is diverging now

    return divtime

center = (-0.743643887037158704752191506114774, 0.131825904205311970493132056385139)
# x0, x1, y0, y1 = -2, 0.8, -1.4, 1.4
# x0, x1, y0, y1 = -1, -0.5, 0, 0.5
x0, x1, y0, y1 = -0.743648, -0.743638, 0.131820, 0.131830
plt.imshow(mandelbrot(x0, x1, y0, y1).T, cmap=plt.cm.jet, origin='lower', extent=(x0, x1, y0, y1))
plt.colorbar()
plt.show()
