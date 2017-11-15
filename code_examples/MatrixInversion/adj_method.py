'''
-------------------------------------------------------------------------------

abstract : find a inverse matrix using the adjugate method

history log :
  2017-02-24  Ki-Hwan Kim  Start
 
-------------------------------------------------------------------------------
'''

from __future__ import print_function, division
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from math import fsum
import numpy as np

from inv_mat_py import inv_f90



def det(mat):
    '''
    Determinant of a NxN matrix
    '''

    N = mat.shape[0]
    submat = np.zeros((N-1,N-1))
    acc = np.zeros(N)

    if N == 1:
        accum = mat[0,0]

    else:
        sgn = 1

        for j in range(N):
            submat[:, :j] = mat[1:, :j]
            submat[:, j:] = mat[1:, j+1:]

            acc[j] = sgn*mat[0,j]*det(submat)
            sgn = -sgn

        accum = fsum(acc)

    return accum



def adjoint(mat):
    '''
    Adjoint of a NxN matrix
    Transpose of the cofactors
    '''

    N = mat.shape[0]
    comat = np.zeros((N,N))
    submat = np.zeros((N-1,N-1))

    for i in range(N):
        for j in range(N):
            submat[:i, :j] = mat[:i, :j]
            submat[:i, j:] = mat[:i, j+1:]
            submat[i:, :j] = mat[i+1:, :j]
            submat[i:, j:] = mat[i+1:, j+1:]

            comat[i,j] = (-1)**(i+j)*det(submat)

    return comat.T



def inv(mat):
    '''
    Inverse matrix using the adjoint method
    '''

    return adjoint(mat)/det(mat)



def main():
    N = 10
    A = np.random.rand(N,N).reshape((N,N), order='F')

    #print(A)
    '''
    aa_equal(np.linalg.det(A), det(A), 15)

    src3x3 = np.array([3,0,2,2,0,-2,0,1,1]).reshape(3,3)
    dst3x3 = np.array([2,2,0,-2,3,10,2,-3,0]).reshape(3,3)
    aa_equal(adjoint(src3x3), dst3x3, 15)

    aa_equal(np.linalg.inv(A), inv(A), 13)
    '''

    print('start ref')
    ref_invA = np.linalg.inv(A)
    print('end ref')

    print('stat f90')
    invA = np.zeros((N,N), order='F')
    inv_f90(1, N, A, invA)
    print('end f90')
    aa_equal(np.linalg.inv(A), invA, 12)



if __name__ == '__main__':
    main()
