#!/usr/bin/env python

import numpy as np
from mpi4py import MPI
from Queue import Queue
from threading import Thread

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

other = {0: 1, 1: 0}[rank]
tag_send = {0: 0, 1: 1}[rank]
tag_recv = {0: 1, 1: 0}[rank]


class QueueTask:
    def __init__(self):
        self.queue = Queue()

        thread = Thread(target=self.work)
        thread.daemon = True
        thread.start()


    def work(self):
        while True:
            func, args = self.queue.get()

            #print 'rank= %d, %s' % (rank, repr(func))
            func(*args)

            self.queue.task_done()


    def enqueue(self, func, args=[]):
        self.queue.put( (func, args) )



def generate_random(iter, arr):
    arr[:] = np.random.rand(*arr.shape)
    np.save('rank%d_random%d' % (rank, iter), arr)


def verify(iter, arr):
    original = np.load('rank%d_random%d.npy' % (other, iter))
    norm = np.linalg.norm(original - arr)
    assert norm == 0, 'rank= %d, random= %d, %g' % (rank, iter, norm)


if __name__ == '__main__':
    nx, ny = 1000, 2000
    iteration = 10

    if rank == 0: 
        print 'nx= %d, ny= %d, iteration= %d' % (nx, ny, iteration)
        print 'initialize...' 

    send_list = []
    recv_list = []
    req_list = []
    qtask_list = []
    for iter in xrange(iteration):
        send = np.zeros((nx, ny))
        recv = np.zeros_like(send)
        send_list.append(send)
        recv_list.append(recv)

        req_list.append( comm.Send_init(send, other, tag_send))# + iter*100) )
        req_list.append( comm.Recv_init(recv, other, tag_recv))# + iter*100) )

        qtask_list.append( QueueTask() )
        if rank == 0: print iter 

    for i in xrange(iteration):
        qtask_list[i].enqueue(generate_random, [i, send_list[i]])
        for req in req_list:
            qtask_list[i].enqueue(req.Start)
        for req in req_list:
            qtask_list[i].enqueue(req.Wait)
        qtask_list[i].enqueue(verify, [i, recv_list[i]])

    for i in xrange(iteration):
        qtask_list[i].queue.join()

    from time import sleep
    sleep(0.5)
    if rank == 0: print 'OK!' 
