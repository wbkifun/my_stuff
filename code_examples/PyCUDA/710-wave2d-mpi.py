#!/usr/bin/env python

import numpy as np
import sys
from matplotlib.pyplot import *
from datetime import datetime
from wave2d_cfunc import update
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()	# Assume size as 2


def exchange_boundary_datas(f):
	if rank == 0:
		comm.Send(f[-2,1:-1], 1, 0)
		comm.Recv(f[-1,1:-1], 1, 1)
	elif rank == 1:
		comm.Recv(f[0,1:-1], 0, 0)
		comm.Send(f[1,1:-1], 0, 1)


nx, ny = 1000, 1000	# total n
tmax = 200
c = np.ones((nx/size,ny),'f')*0.25
f = np.zeros_like(c)
g = np.zeros_like(c)

if rank == 0:
	output = np.ones((nx-2,ny),'f')
	ion()
	imsh = imshow(output.T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.2)
	colorbar()

	t1 = datetime.now()

for tn in range(1,tmax+1):
	if rank == 0: g[400,500] += np.sin(0.1*tn) 

	update(c, f, g)
	exchange_boundary_datas(f)
	update(c, g, f)
	exchange_boundary_datas(g)

	if tn%10 == 0:
		if rank == 1: comm.Send(f[1:,:], 0, 2)
		if rank == 0:
			print "tstep =\t%d/%d (%d %%)\r" % (tn, tmax, float(tn)/tmax*100),
			sys.stdout.flush()
			output[:nx/2-1,:] = f[:-1,:]
			comm.Recv(output[nx/2-1:,:], 1, 2)
			imsh.set_array( np.sqrt(output.T**2) )
			draw()
			#savefig('./png/%.5d.png' % tn) 

if rank == 0:
	print ''
	print datetime.now() - t1
