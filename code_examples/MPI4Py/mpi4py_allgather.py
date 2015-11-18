from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
myrank = comm.Get_rank()


d = (myrank+1)**2
ds = comm.allgather(d)

print myrank, ds
