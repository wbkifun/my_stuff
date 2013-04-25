#!/usr/bin/env python

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


nx, ny = 10000, 1000

if rank == 0:
    arr = np.random.rand(nx, ny)
    recv = np.zeros_like(arr)

    req_send = comm.Send_init(arr, 2, tag=0)
    req_recv = comm.Recv_init(recv, 1, tag=1)

    req_send.Start()
    req_recv.Start()
    req_send.Wait()
    req_recv.Wait()

    norm = np.linalg.norm(arr - recv)
    assert norm == 0, 'rank= %d, %g' % (rank, norm)

elif rank == 1:
    tmp = np.zeros((nx, ny))

    comm.Recv(tmp, 0, tag=0)
    comm.Send(tmp, 0, tag=1)
