#!/usr/bin/env python

import numpy as np

c = np.ones((1000,1000),'f')*0.25
f = np.zeros_like(c)
g = np.zeros_like(c)

ii = (slice(1,-1), slice(1,-1))
for tn in range(1,501):
	g[400,500] += np.sin(0.1*tn) 
	f[ii] = c[ii]*(g[2:,1:-1] + g[:-2,1:-1] + g[1:-1,2:] + g[1:-1,:-2] - 4*g[ii]) + 2*g[ii] - f[ii]
	g[ii] = c[ii]*(f[2:,1:-1] + f[:-2,1:-1] + f[1:-1,2:] + f[1:-1,:-2] - 4*f[ii]) + 2*f[ii] - g[ii]
