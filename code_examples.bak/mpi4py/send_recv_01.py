#!/usr/bin/env python

from __future__ import division
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


n = 10

if rank == 0:
    a = np.random.rand(n)
    b = np.zeros_like(a)

    comm.Send(a, 1, 10)
    print 'rank', rank, 'Send OK!'
    comm.Recv(b, 1, 20)
    print 'rank', rank, 'Recv OK!'

    print 'norm', np.linalg.norm(a - b)

elif rank == 1:
    a = np.zeros(n)

    comm.Recv(a, 0, 10)
    print 'rank', rank, 'Recv OK!'
    comm.Send(a, 0, 20)
    print 'rank', rank, 'Send OK!'
