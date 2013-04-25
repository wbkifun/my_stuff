#!/usr/bin/env python

from __future__ import division
import numpy as np
import parallel as par


rank, size, comm, request = par.rank, par.size, par.comm, par.MPI.Request
tag = 10


def exchange(arr, arr2, s_sche, r_sche):
    req_send_list = []
    for dest, (i0, i1) in s_sche.items():
        req = comm.Isend(arr[i0:i1], dest, tag)
        req_send_list.append(req)

    req_recv_list = []
    for src, (i0, i1) in r_sche.items():
        req = comm.Irecv(arr2[i0:i1], src, tag)
        req_recv_list.append(req)

    request.Waitall(req_send_list)
    request.Waitall(req_recv_list)

    for i0, i1 in r_sche.values():
        arr[i0:i1] = arr2[i0:i1]



if __name__ == '__main__':
    n = 2
    arr = np.ones(n * size) * rank
    arr2 = np.zeros_like(arr)

    s_sche = { \
            0:{1:(0,n), 2:(0,n), 3:(0,n)}, \
            1:{0:(n,2*n), 2:(n,2*n), 3:(n,2*n)}, \
            2:{0:(2*n,3*n), 1:(2*n,3*n), 3:(2*n,3*n)}, \
            3:{0:(3*n,4*n), 1:(3*n,4*n), 2:(3*n,4*n)} }[rank]
    r_sche = { \
            0:{1:(n,2*n), 2:(2*n,3*n), 3:(3*n,4*n)}, \
            1:{0:(0,n), 2:(2*n,3*n), 3:(3*n,4*n)}, \
            2:{0:(0,n), 1:(n,2*n), 3:(3*n,4*n)}, \
            3:{0:(0,n), 1:(n,2*n), 2:(2*n,3*n)} }[rank]

    exchange(arr, arr2, s_sche, r_sche)

    print rank, arr
