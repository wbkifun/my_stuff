from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


assert nproc == 4


if rank == 0:
    my_list = [1,2,3]

elif rank == 1:
    my_list = [4,5,6,7]

elif rank == 2:
    my_list = [8]

elif rank == 3:
    my_list = [9,10]


print 'rank=%d, my_list=%s' % (rank, my_list)
comm.Barrier()


gather_list = comm.gather(my_list, root=0)
print 'rank=%d, my_list=%s' % (rank, gather_list)
