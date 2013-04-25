#!/usr/bin/env python

'''
import scipy as sc
import matplotlib.pyplot as pl

x = sc.arange(-5, 5, 0.1, 'f')
y = sc.exp( -( x**2 ) )

pl.plot(y)
#pl.xticks( sc.arange(0, len(x), 10), sc.arange(-5, 5) )
#pl.ylim( 0, 1.1 )
pl.show()
'''

import numpy
import matplotlib.pyplot as pyplot

x = numpy.arange(-5, 5, 0.1, 'f')
y = numpy.exp( -( x**2 ) )

pyplot.plot(x, y, linewidth=3)
pyplot.savefig('./png100/1d-gaussian-01.png')
pyplot.show()
