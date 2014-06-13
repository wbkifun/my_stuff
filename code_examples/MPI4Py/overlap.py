from __future__ import division
import numpy as np
from math import fsum
from datetime import datetime
from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nprocs = comm.Get_size()



def job(N=12000):
    ret = 0
    for i in xrange(N):
        for j in xrange(N):
            ret += (i-j)

    return ret



def communicate_block(a, b):
    if myrank == 0:
        comm.Send(a, dest=1, tag=0)
        comm.Recv(b, source=1, tag=1)

    elif myrank == 1:
        comm.Recv(b, source=0, tag=0)
        comm.Send(a, dest=0, tag=1)



def communicate_nonblock(a, b):
    opponent = {0:1, 1:0}[myrank]
    tag_send = {0:0, 1:1}[myrank]
    tag_recv = {0:1, 1:0}[myrank]

    req_send = comm.Send_init(a, opponent, tag_send)
    req_recv = comm.Recv_init(b, opponent, tag_recv)

    req_send.Start()
    req_recv.Start()

    return req_send, req_recv



def job_core(comm_type):
    Ncomm = 6e8
    comm_a = np.arange(Ncomm)
    comm_b = np.empty(Ncomm)


    t0 = datetime.now()

    comp_a = job()
    comp_b = job()

    dt_comp = datetime.now() - t0
    t1 = datetime.now()

    if comm_type == 'block':
        communicate_block(comm_a, comm_b)

    elif comm_type == 'nonblock':
        req_send, req_recv = communicate_nonblock(comm_a, comm_b)
        req_send.Wait()
        req_recv.Wait()

    dt_comm = datetime.now() - t1
    dt_tot = datetime.now() - t0

    
    return (comp_a, comp_b), (comm_a, comm_b), (dt_comp, dt_comm, dt_tot)



def job_core_overlap():
    Ncomm = 6e8
    comm_a = np.arange(Ncomm)
    comm_b = np.empty(Ncomm)


    t0 = datetime.now()
    comp_a = job()

    req_send, req_recv = communicate_nonblock(comm_a, comm_b)

    comp_b = job()

    req_send.Wait()
    req_recv.Wait()

    dt_tot = datetime.now() - t0

    
    return (comp_a, comp_b), (comm_a, comm_b), (None, None, dt_tot)



def run(comm_type):
    #--------------------------------------------------
    # computation and communication
    #--------------------------------------------------
    if comm_type == 'block':
        comp_ab, comm_ab, dts = job_core('block')

    elif comm_type == 'nonblock':
        comp_ab, comm_ab, dts = job_core('nonblock')

    elif comm_type == 'overlap':
        comp_ab, comm_ab, dts = job_core_overlap()


    #--------------------------------------------------
    # verify
    #--------------------------------------------------
    diff1 = comp_ab[0] - comp_ab[1]
    assert diff1==0, \
            'computation error: rank=%d, a-b=%d'%(myrank, diff1)

    diff2 = comm_ab[0].sum()-comm_ab[1].sum()
    assert diff2==0, 'communication error: rank=%d, a-b=%d'%(myrank, diff2)


    #--------------------------------------------------
    # print
    #--------------------------------------------------
    if myrank == 0:
        dt_comp, dt_comm, dt_tot = dts

        print '<%s>'%(comm_type)
        print 'computation time:\t%s'%(dt_comp)
        print 'communication time:\t%s'%(dt_comm)
        print 'total time:\t\t%s'%(dt_tot)
        print ''



if __name__ == '__main__':
    run('block')
    run('nonblock')
    run('overlap')
