#!/usr/bin/env python

import numpy as np

c = np.ones((1000,1000),'f')*0.25
f = np.zeros_like(c)
g = np.zeros_like(c)

import sys
from matplotlib.pyplot import *
ion()
imsh = imshow(np.ones_like(c), cmap=cm.hot, origin='lower', vmin=0, vmax=0.2)
colorbar()

ii = (slice(1,-1), slice(1,-1))
tmax = 50

from datetime import datetime
t1 = datetime.now()

for tn in range(1,tmax+1):
	g[400,500] += np.sin(0.1*tn) 
	f[ii] = c[ii]*(g[2:,1:-1] + g[:-2,1:-1] + g[1:-1,2:] + g[1:-1,:-2] - 4*g[ii]) + 2*g[ii] - f[ii]
	g[ii] = c[ii]*(f[2:,1:-1] + f[:-2,1:-1] + f[1:-1,2:] + f[1:-1,:-2] - 4*f[ii]) + 2*f[ii] - g[ii]

	if tn%10 == 0:
		print "tstep =\t%d/%d (%d %%)\r" % (tn, tmax, float(tn)/tmax*100),
		sys.stdout.flush()
		imsh.set_array( np.sqrt(f.T**2) )
		draw()
		#savefig('./png/%.5d.png' % tn) 

print ''
print datetime.now() - t1
