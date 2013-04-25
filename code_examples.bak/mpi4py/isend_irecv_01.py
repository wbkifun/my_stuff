#!/usr/bin/env python

from __future__ import division
import numpy as np
import parallel as par


rank, size, comm = par.rank, par.size, par.comm
n = 2
tag = 10

arr = np.ones(n * size) * rank

dests = range(rank) + range(rank+1, size)
srcs = range(rank) + range(rank+1, size)

req_send_list = []
for dest in dests:
    req = comm.Isend(arr[rank*n:(rank+1)*n], dest, tag)
    req_send_list.append(req)

req_recv_list = []
for src in srcs:
    req = comm.Irecv(arr[src*n:(src+1)*n], src, tag)
    req_recv_list.append(req)


par.request.Waitall(req_send_list)
par.request.Waitall(req_recv_list)

print par.rank, arr
