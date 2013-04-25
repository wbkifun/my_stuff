#!/usr/bin/env python

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


other = {0: 1, 1: 0}[rank]
tag_send = {0: 0, 1: 1}[rank]
tag_recv = {0: 1, 1: 0}[rank]

nx, ny = 1000, 2000
iteration = 10

if rank == 0: 
    print 'nx= %d, ny= %d, iteration= %d' % (nx, ny, iteration)
    print 'initialize...' 

send_list = []
recv_list = []
req_list = []
for iter in xrange(iteration):
    send = np.random.rand(nx, ny)
    recv = np.zeros_like(send)
    send_list.append(send)
    recv_list.append(recv)
    np.save('rank%d_random%d' % (rank, iter), send)

    req_list.append( comm.Send_init(send, other, tag_send + iter*100) )
    req_list.append( comm.Recv_init(recv, other, tag_recv + iter*100) )
    if rank == 0: print iter 


if rank == 0: print 'Start...' 
for req in req_list: req.Start()

if rank == 0: print 'Wait...' 
for req in req_list: req.Wait()

if rank == 0: print 'Verify...' 
for iter, recv in enumerate(recv_list):
    original = np.load('rank%d_random%d.npy' % (other, iter))
    norm = np.linalg.norm(original - recv)
    assert norm == 0, 'rank= %d, random= %d, %g' % (rank, iter, norm)

if rank == 0: print 'OK!' 
