#------------------------------------------------------------------------------
# filename  : nqueens_single.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Development Team, KIAPS
# update    : 2014.7.29     start
#
#
# description: 
#   Solve the NQueens problem for the NxN board
#   originally C code by Paulo Marques
#   refer to http://www.drdobbs.com/task-farming-the-message-passing-interf/184405430
#
# class:
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data



def n_queens(size):
    board = np.zeros(size, np.int32)
    n_solutions = place_queen(0, board)

    return n_solutions



def place_queen(col, board):
    size = len(board)

    n_solutions = 0

    print '='*47
    print 'col', col

    # Try to place a queen in each line of <column>
    for row in xrange(size):
        print '-'*20
        print 'row', row
        board[col] = row

        # Check if this board is still a solution
        is_sol = True

        for j in xrange(col-1,-1,-1):
            print ''
            print 'col, j', col, j
            print board[col], (board[j], board[j]-(col-j), board[j]+(col-j))
            if board[col] in [board[j], board[j]-(col-j), board[j]+(col-j)]:
                is_sol = False
                break

        if is_sol:
            print 'ok'
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
    print n_queens(4)

    '''
    from datetime import datetime
    for n in xrange(2,20):
        t1 = datetime.now()
        print n_queens(n)
        t2 = datetime.now()
        print 'n=%d'%n, t2-t1
    '''
