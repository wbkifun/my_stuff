#!/usr/bin/env python

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


nx, ny = 1000, 1000

send = np.random.rand(nx, ny)
recv = np.zeros_like(send)
np.save('rank%d_random%d' % (rank, 0), send)

other = {0: 1, 1: 0}[rank]
tag_send = {0: 0, 1: 1}[rank]
tag_recv = {0: 1, 1: 0}[rank]

req_send = comm.Send_init(send, other, tag_send)
req_recv = comm.Recv_init(recv, other, tag_recv)

req_send.Start()
req_recv.Start()
req_send.Wait()
req_recv.Wait()

original = np.load('rank%d_random%d.npy' % (other, 0))
norm = np.linalg.norm(original - recv)
assert norm == 0, 'rank= %d, %g' % (rank, norm)
