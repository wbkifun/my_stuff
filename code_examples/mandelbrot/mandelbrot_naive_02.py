import matplotlib.pyplot as plt



def mandelbrot(x0, x1, y0, y1, res=400, maxiter=100):
    '''Returns an image of the Mandelbrot fractal'''
    
    x, y = numpy.ogrid[x0:x1:res*1j, y0:y1:res*1j]
    x = numpy.linspace(x0,x1,res)
    y = numpy.linspace(y0,y1,res)
    c = x + y*1j
    z = numpy.zeros_like(c)
    divtime = numpy.ones(c.shape, dtype=int) * maxiter

    for n in xrange(maxiter):
        for i in xrange(res):
            for j in xrange(res):
                z[i,j] = z[i,j]**2 + c[i,j]
                if abs(z[i,j]) > 2: 
                    z[i,j] = 2
                    if divtime[i,j] == maxiter:
                        divtime[i,j] = n
                '''
                diverge = numpy.abs(z) > 2                 # who is diverging
                z[diverge] = 2                             # avoid diverging too much
                divtime[diverge & (divtime==maxiter)] = n  # note who is diverging now
                '''

    return divtime



x0, y0 = (-0.743643887037158704752191506114774, 0.131825904205311970493132056385139)
dx, dy = 0.0001, 0.0001
x1, x2, y1, y2 = x0-dx, x0+dx, y0-dy, y0+dy
plt.imshow(mandelbrot(x1, x2, y1, y2).T, cmap=plt.cm.jet, origin='lower')#, extent=(x1, x2, y1, y2))
plt.colorbar()
plt.show()
