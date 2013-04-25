#!/usr/bin/env python

import scipy as sc
import matplotlib.pyplot as pl
import matplotlib.cm as cm

x = sc.arange(-3,3,0.01,'f')
y = x[:,sc.newaxis]
f = sc.exp(-(x**2)-(y**2))
df = (f[1:,1:]-f[:-1,:-1])/0.01

pl.subplot(2,2,1)
pl.plot(x, f[:,300])
pl.subplot(2,2,2)
pl.plot(x[:-1], df[:,300])
pl.subplot(2,2,3)
pl.imshow(f, cmap=cm.hot, origin='lower', extent=[-3,3,-3,3])
pl.colorbar()
pl.subplot(2,2,4)
pl.imshow(df, cmap=cm.jet, origin='lower', extent=[-3,3,-3,3])
pl.colorbar()

pl.savefig('./png100/1d-gaussian-sin.png')
pl.show()
