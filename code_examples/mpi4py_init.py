from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() + 1
nproc = comm.Get_size()


print 'size= %d, rank= %d' % (nproc, rank)
