from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() + 1
nproc = comm.Get_size()


print("size= {}, rank= {}".format(nproc, rank))
