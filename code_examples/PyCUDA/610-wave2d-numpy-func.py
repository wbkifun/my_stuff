#!/usr/bin/env python

import numpy as np
import sys
from matplotlib.pyplot import *
from datetime import datetime


ii = (slice(1,-1), slice(1,-1))
def update(c, f, g):
	f[ii] = c[ii]*(g[2:,1:-1] + g[:-2,1:-1] + g[1:-1,2:] + g[1:-1,:-2] - 4*g[ii]) + 2*g[ii] - f[ii]


nx, ny = 1000, 1000
tmax = 100
c = np.ones((nx,ny),'f')*0.25
f = np.zeros_like(c)
g = np.zeros_like(c)

ion()
imsh = imshow(np.ones_like(c), cmap=cm.hot, origin='lower', vmin=0, vmax=0.2)
colorbar()

t1 = datetime.now()
for tn in range(1,tmax+1):
	g[400,500] += np.sin(0.1*tn) 
	update(c, f, g)
	update(c, g, f)

	if tn%10 == 0:
		print "tstep =\t%d/%d (%d %%)\r" % (tn, tmax, float(tn)/tmax*100),
		sys.stdout.flush()
		imsh.set_array( np.sqrt(f.T**2) )
		draw()
		#savefig('./png/%.5d.png' % tn) 

print ''
print datetime.now() - t1
