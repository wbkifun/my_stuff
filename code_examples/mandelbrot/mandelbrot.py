from __future__ import division
import numpy
import matplotlib.pyplot as plt


from core_fortran import diverge


def mandelbrot(x0, x1, y0, y1, res=400, maxiter=500):
    '''Returns an image of the Mandelbrot fractal'''
    
    c = numpy.zeros((res,res), dtype=numpy.complex128, order='F')
    z = numpy.zeros_like(c, order='F')
    numdiv = numpy.ones(c.shape, dtype=numpy.int32, order='F') * maxiter

    x, y = numpy.ogrid[x0:x1:res*1j, y0:y1:res*1j]
    c[:] = x + y*1j

    diverge(c, z, numdiv, maxiter)

    return numdiv



x0, y0 = (-0.743643887037158704752191506114774, 0.131825904205311970493132056385139)
dx, dy = 0.0001, 0.0001
x1, x2, y1, y2 = x0-dx, x0+dx, y0-dy, y0+dy
plt.imshow(mandelbrot(x1, x2, y1, y2).T, cmap=plt.cm.jet, origin='lower')#, extent=(x1, x2, y1, y2))
plt.colorbar()
plt.show()

