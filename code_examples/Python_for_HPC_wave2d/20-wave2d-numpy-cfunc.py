#!/usr/bin/env python
#------------------------------------------------------------------------------
# File Name : wave2d-numpy-cfunc.py
#
# Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
# 
# Written date :	2010. 6. 17
# Modify date :		
#
# Copyright : GNU GPL
#
# Description : 
# Simulation for the 2-dimensional wave equations with simple FD (Finite-Difference) scheme
#
# These are educational codes to study Python programming for high performance computing.
# Step 1: Using numpy arrays
# Step 2-4: Combining with C function (C-level Optimizations)
#------------------------------------------------------------------------------

import numpy as np
import sys
from matplotlib.pyplot import *
from datetime import datetime
from wave2d_cfunc import update

nx, ny = 1000, 1000
tmax = 200
c = np.ones((nx,ny),'f')*0.25
f = np.zeros_like(c)
g = np.zeros_like(c)

# To plot using matplotlib
ion()
imsh = imshow(np.ones_like(c).T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.2)
colorbar()

# To measure the execution time
t1 = datetime.now()

# Main loop for time evolution
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

print '\n', datetime.now() - t1
