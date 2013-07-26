from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


assert nproc == 4


if rank == 0:
    my_dict = {'a':1, 'b':2, 'c':3}

elif rank == 1:
    my_dict = {'d':4, 'e':5, 'f':6, 'g':7}

elif rank == 2:
    my_dict = {'h':8}

elif rank == 3:
    my_dict = {'i':9, 'j':10}


print 'rank=%d, my_dict=%s' % (rank, my_dict)
comm.Barrier()


gather_list = comm.gather(my_dict, root=0)
print 'rank=%d, gather_list=%s' % (rank, gather_list)


if rank == 0:
    united_dict = dict()
    for sub_dict in gather_list:
        united_dict.update(sub_dict)

    print 'rank=%d, united_dict=%s' % (rank, united_dict)
