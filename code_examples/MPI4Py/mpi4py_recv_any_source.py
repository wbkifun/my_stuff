from __future__ import division
from random import randint
from time import sleep
from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()



if myrank == 0:
    for seq in xrange(10):
        rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
        comm.send(seq, dest=rank, tag=10)

    print 'master: all job is sent.'

    for proc in xrange(nprocs-1):
        rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
        comm.send('quit', dest=rank, tag=10)


else:
    while True:
        comm.send(myrank, dest=0, tag=0)
        jobid = comm.recv(source=0, tag=10)

        if jobid == 'quit':
            break

        else:
            sleep_time = randint(1,3)
            print 'rank=%d, job_id=%d, sleep_time=%d'%(myrank, jobid, sleep_time)

            sleep(sleep_time)


    print 'rank=%d, quited.' % myrank
