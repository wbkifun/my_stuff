#!/usr/bin/env python

from __future__ import division
import numpy as np
import parallel as par


rank, size, comm, request = par.rank, par.size, par.comm, par.MPI.Request
tag = 10



class Schedule(object):
    def __init__(self):
        self.length = {}
        self.i0_send = {}
        self.i0_recv = {}



def exchange(buf, sche):
    tmp = np.zeros_like(buf)

    req_send_list = []
    for dest, i0 in sche.i0_send.items():
        i1 = i0 + sche.length[dest]
        req = comm.Isend(buf[i0:i1], dest, tag)
        req_send_list.append(req)

    req_recv_list = []
    for src, i0 in sche.i0_recv.items():
        i1 = i0 + sche.length[src]
        req = comm.Irecv(tmp[i0:i1], src, tag)
        req_recv_list.append(req)

    request.Waitall(req_send_list)
    request.Waitall(req_recv_list)

    for src, i0 in sche.i0_recv.items():
        i1 = i0 + sche.length[src]
        buf[i0:i1] = tmp[i0:i1]




if __name__ == '__main__':
    n = 2
    buf = np.ones(n * size) * rank

    sche = Schedule()
    for target in xrange(size):
        sche.length[target] = n
        sche.i0_send[target] = rank*n
        sche.i0_recv[target] = target*n
            
    if rank == 0: print 'before'
    par.comm.barrier()
    print rank, buf

    exchange(buf, sche)

    if rank == 0: print 'after'
    par.comm.barrier()
    print rank, buf
