#!/usr/bin/env python

import scipy
import matplotlib.pyplot as pyplot

x = scipy.arange(-5,5,0.01,'f')
a = scipy.exp(-(x**2)/2)
b = scipy.sin(5*x)

c = a*b

pyplot.plot(x, a, x, b)
pyplot.plot(x, c, linewidth=3)
pyplot.savefig('./png100/1d-gaussian-sin.png')
pyplot.show()
