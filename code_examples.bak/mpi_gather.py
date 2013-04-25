#!/usr/bin/env python

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ta = 1
a = np.ones(10, 'f')*rank

if( rank==0 ):
	ta = np.zeros((3*10), 'f')

comm.Gather(a, ta)

if( rank==0 ):
	print ta
