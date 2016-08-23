from __future__ import division
import numpy as np
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
myrank = mpi_comm.Get_rank()
nprocs = mpi_comm.Get_size()



class OnMaster(object):
    def __init__(self, func, *args):
        self.func = func
        self.args = args


    def __call__(self):
        if myrank == 0:
            self.func(*self.args)




def onlyMaster(func):
    def new_func(*args):
        if myrank == 0:
            func(*args)

    return new_func




class TestClass(object):
    def __init__(self):
        pass


    def print_myrank_all(self):
        print myrank


    @onlyMaster
    def print_myrank_master(self):
        print myrank




if __name__ == '__main__':
    tc = TestClass()
    tc.print_myrank_all()
    tc.print_myrank_master()
