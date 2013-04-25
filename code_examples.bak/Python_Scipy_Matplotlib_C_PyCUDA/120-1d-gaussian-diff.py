#!/usr/bin/env python

import numpy
import matplotlib.pyplot as pyplot

x = numpy.arange(-5, 5, 0.1, 'f')
y = numpy.exp( -( x**2 ) )
#y2 = numpy.exp( -( x[:]**2 ) )*(-2*x[:])
dy = ( y[1:] - y[:-1] )/0.1

#pyplot.plot(x[1:], dy, x, y2, linewidth=3)
pyplot.plot(x, y, x[1:], dy, linewidth=3)
pyplot.savefig('./png100/1d-gaussian-diff.png')
pyplot.show()
