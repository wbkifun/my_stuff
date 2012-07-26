#!/usr/bin/env python

import scipy as sc
from matplotlib.pyplot import *

c0 = 0.25
c = sc.ones((1000,1000),'f')*c0
f = sc.zeros_like(c)
g = sc.zeros_like(c)

c[600:620,:] = 0
c[600:620,350:400] = c0
c[600:620,600:650] = c0

i = slice(1,-1)
ii = (i,i)

ion()
imsh = imshow(sc.ones_like(c), cmap=cm.hot, origin='lower', vmin=0, vmax=0.2)
colorbar()
for tstep in xrange(1000):
	g[400,500] = sc.sin(0.1*tstep)
	#f[1:-1,1:-1] = 0.25*(g[2:,1:-1]+g[:-2,1:-1]+g[1:-1,2:]+g[1:-1,:-2]-4*g[1:-1,1:-1])+2*g[1:-1,1:-1]-f[1:-1,1:-1]
	#g[1:-1,1:-1] = 0.25*(f[2:,1:-1]+f[:-2,1:-1]+f[1:-1,2:]+f[1:-1,:-2]-4*f[1:-1,1:-1])+2*f[1:-1,1:-1]-g[1:-1,1:-1]
	f[ii] = c[ii]*(g[2:,i]+g[:-2,i]+g[i,2:]+g[i,:-2]-4*g[ii])+2*g[ii]-f[ii]
	g[ii] = c[ii]*(f[2:,i]+f[:-2,i]+f[i,2:]+f[i,:-2]-4*f[ii])+2*f[ii]-g[ii]

	if tstep%10 == 0:
		imsh.set_array( sc.sqrt(f.T**2) )
		draw()
		#savefig(png_str) 
