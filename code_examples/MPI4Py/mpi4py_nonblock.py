from __future__ import division
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
myrank = comm.Get_rank()


if myrank == 0:
    req_list = list()

    for target_rank in xrange(1,nproc):
        data = {'a':np.array([1,2,3]), 'b':np.array([1,2,3])*target_rank}
        req = comm.isend(data, dest=target_rank, tag=1)
        req_list.append(req)
    
    for req in req_list: req.wait()

else:
    data = comm.recv(source=0, tag=1)
    #req = comm.irecv(dest=0, tag=1)
    #data = req.wait()


    print myrank, data
