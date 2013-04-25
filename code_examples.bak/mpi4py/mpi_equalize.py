from __future__ import division
from mpi4py import MPI
import numpy


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def equalize(nx):
    a_local = numpy.ones(nx, int) * rank
    a_global = numpy.ones(nx*size, int) * rank

    dests = range(size)
    dests.remove(rank)
    reqs = []
    for dest in dests:
        reqs.append( comm.Isend(a_local, dest) )
        reqs.append( comm.Irecv(a_global[dest*nx:(dest+1)*nx], dest) )

    MPI.Request.Waitall(reqs)

    return a_global



if __name__ == '__main__':
    nx = 4
    a = equalize(nx)
    a_ref = numpy.zeros(nx*size, int)
    for dest in xrange(size):
        a_ref[dest*nx:(dest+1)*nx] = dest

    assert numpy.linalg.norm(a - a_ref) == 0, 'assert in rank %d' % rank
