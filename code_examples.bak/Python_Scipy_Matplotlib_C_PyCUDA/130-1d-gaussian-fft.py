#!/usr/bin/env python

import numpy, scipy
import matplotlib.pyplot as pyplot

dx = 0.01
N = 1000
x = numpy.arange(-5, 5, dx, 'f')
y = numpy.exp( -( x**2 )/2 )
#y2 = numpy.exp( -( x[:]**2 ) )*(-x[:])
#dy = ( y[1:] - y[:-1] )/0.1
fft_y = scipy.fft( y )
fft_y2 = numpy.concatenate( (fft_y[N/2:],fft_y[:N/2]) )

#pyplot.plot(x, y, linewidth=3)
#pyplot.plot(x[1:], dy, x, y2, linewidth=3)
#pyplot.plot(x, y, x[1:], dy, linewidth=3)
#pyplot.plot(x, fft_y.real, x, fft_y.imag, x, abs(fft_y), linewidth=3)
#pyplot.plot(x, fft_y2.real, x, fft_y2.imag, x, abs(fft_y2))
k = numpy.arange(-1/(2*dx),1/(2*dx),1/(len(x)*dx),'f') 
pyplot.plot(k, abs(fft_y2), linewidth=3)
pyplot.savefig('./png100/1d-gaussian-fft.png')
pyplot.show()
