#!/usr/bin/env python
#---------------------------------------------------------------------------
# File Name : wave2d_numpy.py
#
# Author : Ki-Hwan Kim (wbkifun@nate.com)
# 
# Written date :	2010. 6. 17
# Modify date :		
#
# Copyright : GNU GPL
#
# Description : 
# Simulation for the 2-dimensional wave equations with simple FD (Finite-Difference) scheme
#
# These are educational codes to study python programming for high performance computing.
# Step 1: Using numpy arrays
#---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def update(c, f, g):
	f[1:-1,1:-1] = c[1:-1,1:-1]*(g[2:,1:-1] + g[:-2,1:-1] + g[1:-1,2:] + g[1:-1,:-2] - 4*g[1:-1,1:-1]) + 2*g[1:-1,1:-1] - f[1:-1,1:-1]


nx, ny = 1000, 800
tmax = 500
f = np.zeros((nx,ny), dtype=np.float32)
g = np.zeros_like(f)
c = np.ones_like(f) * 0.25

# To plot using matplotlib
plt.ion()
img = plt.imshow(f.T, origin='lower', vmin=-0.3, vmax=0.3)
plt.colorbar()

# Main loop for time evolution
for tn in xrange(1,tmax+1):
	g[400,300] += np.sin(0.1*tn) 
	update(c, f, g)
	update(c, g, f)

	if tn%10 == 0:
		print("tstep =\t%d/%d (%d %%)" % (tn, tmax, float(tn)/tmax*100))
		img.set_array(f.T)
		plt.draw()
		#savefig('./png/%.5d.png' % tn) 
