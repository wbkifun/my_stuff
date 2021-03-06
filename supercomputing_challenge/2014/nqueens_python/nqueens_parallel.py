#------------------------------------------------------------------------------
# filename  : nqueens_parallel.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Development Team, KIAPS
# update    : 2014.7.29     start
#             2014.8.27     parallel version
#
#
# description: 
#   Solve the NQueens problem for the NxN board
#   originally C code by Paulo Marques
#   refer to http://www.drdobbs.com/task-farming-the-message-passing-interf/184405430
#
# subroutine:
#   n_queens()
#   place_queen()
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
myrank = mpi_comm.Get_rank()
nprocs = mpi_comm.Get_size()



def master(size):
    board = np.zeros(size, np.int32)
    n_solutions = master_place_queen(0, board)

def n_queens(size):

    return n_solutions



def place_queen(col, board):
    size = len(board)

    n_solutions = 0

    # Try to place a queen in each line of <column>
    for i in xrange(size):
        board[col] = i

        # Check if this board is still a solution
        is_sol = True

        for j in xrange(col-1,-1,-1):
            if board[col] in [board[j], board[j]-(col-j), board[j]+(col-j)]:
                is_sol = False
                break

        if is_sol:
            if col == size-1:
                # If this is the last column, printout the solution
                n_solutions += 1
                #print 'solution', board

            else:
                # The board is not complete.
                # Try to place the queens on the next level, using the current board
                n_solutions += place_queen(col+1,board)

    return n_solutions




if __name__ == '__main__':
    from datetime import datetime
    for n in xrange(2,20):
        t1 = datetime.now()
        print n_queens(n)
        t2 = datetime.now()
        print 'n=%d'%n, t2-t1
