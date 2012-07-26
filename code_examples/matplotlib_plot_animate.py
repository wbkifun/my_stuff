#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import *

x = np.arange(10)
a = np.zeros(10, 'f')
a[:] = x[:]

ion()
line, = plot(x,a)
for i in range(10):
	#line.set_ydata(a[:]+i*0.1)

	a[:] += 0.1
	line.set_ydata(a[:] + 0.00001)
	draw()
