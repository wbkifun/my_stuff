from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(x0, x1, y0, y1, divtime, res=400, maxiter=20):
    '''Returns an image of the Mandelbrot fractal'''
    
    x, y = np.ogrid[x0:x1:res*1j, y0:y1:res*1j]
    c = x + y*1j
    z = 0
    divtime[:] = maxiter

    for i in xrange(maxiter):
        z = z**2 + c
        diverge = np.abs(z) > 2                    # who is diverging
        z[diverge] = 2                             # avoid diverging too much
        divtime[diverge & (divtime==maxiter)] = i  # note who is diverging now


center = (-0.743643887037158704752191506114774, 0.131825904205311970493132056385139)
#x0, x1, y0, y1 = -2, 0.8, -1.4, 1.4
x0, x1, y0, y1 = center[0]-1, center[0]+1, center[1]-1, center[1]+1
res, maxiter = 400, 20
scale = 0.2

plt.ion()
divtime = np.ones((res,res), dtype=int)
imag = plt.imshow(divtime.T, cmap=plt.cm.jet, origin='lower', extent=(x0, x1, y0, y1), vmin=0, vmax=maxiter)
plt.xticks([])
plt.yticks([])

for i in xrange(20):
    x0 += abs(center[0] - x0) * scale
    x1 -= abs(center[0] - x1) * scale
    y0 += abs(center[1] - y0) * scale
    y1 -= abs(y1 - center[1]) * scale
    maxiter=int(maxiter/scale)
    mandelbrot(x0, x1, y0, y1, divtime, res, maxiter)
    imag.set_data(divtime.T)
    plt.draw()
