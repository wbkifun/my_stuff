#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt



nx = 100
x = np.linspace(-10, 10, nx)
f = np.exp(-x**2/10)        # Gaussian function


# check the initial state
plt.ion()                   # plot interactive on
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Moving 1D Gaussian')

[line] = ax.plot(x,f)      # create the line object
#plt.show(True)             # show the static plot


# animate
dx = x[1] - x[0]
for tstep in xrange(50):
    f[:] = np.exp(-(x-dx*tstep)**2/10)  # update the function

    line.set_ydata(f)               # update the line object
    plt.draw()                      # redraw
