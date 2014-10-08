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
    solutions = place_queen(0, board)

    return solutions



def place_queen(col, board):
    size = len(board)

    solutions = list()

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
                solutions.append(board.copy())

            else:
                # The board is not complete.
                # Try to place the queens on the next level, using the current board
                solutions += place_queen(col+1,board)

    return solutions



def plot_board(board):
    n = len(board)
    f = np.ones((n,n),np.int32)
    for i, j in enumerate(board):
        f[i,j] = 0

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    plt.imshow(f, cmap=plt.cm.gray, interpolation='nearest')

    for i in xrange(n-1):
        plt.plot([-0.5,n-0.5], [i+0.5,i+0.5], 'k-')
        plt.plot([i+0.5,i+0.5], [-0.5,n-0.5], 'k-')

    ax.set_title('%d-Queens'%n)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlim(-0.5,n-0.5)
    ax.set_ylim(-0.5,n-0.5)
    plt.tight_layout(pad=2)
    plt.show(True)




if __name__ == '__main__':
    solutions = n_queens(5)
    print 'number of solutions:', len(solutions)
    plot_board(solutions[9])
