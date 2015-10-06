#------------------------------------------------------------------------------
# filename  : test_cube_partition.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.8   revise
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_set_factor_list():
    '''
    factor_list: ne=30, nproc=1, 6*2*2, 6*3*3
    '''
    from cube_partition import CubePartition

    cube = CubePartition(ne=2*3*5, nproc=1)
    a_equal(cube.factor_list, [2,3,5])

    cube = CubePartition(ne=2*3*5, nproc=6*2*2)
    a_equal(cube.factor_list, [3,5,2])

    cube = CubePartition(ne=2*3*5, nproc=6*3*3)
    a_equal(cube.factor_list, [2,5,3])




@raises(AssertionError)
def test_set_factor_list_exception():
    '''
    factor_list exception: ne=3*7, nproc=1
    '''
    from cube_partition import CubePartition

    cube = CubePartition(ne=3*7, nproc=1)




def test_set_sfc():
    '''
    sfc: ne=4, 6, 10
    '''
    from cube_partition import CubePartition

    cube = CubePartition(ne=2*2, nproc=1)
    expect = np.array([ \
            [ 1, 4, 5, 6], \
            [ 2, 3, 8, 7], \
            [15,14, 9,10], \
            [16,13,12,11]], 'i4')
    a_equal(cube.sfc, expect)


    cube = CubePartition(ne=2*3, nproc=1)
    expect = np.array([ \
            [ 1, 2,15,16,17,18], \
            [ 4, 3,14,13,20,19], \
            [ 5, 8, 9,12,21,22], \
            [ 6, 7,10,11,24,23], \
            [35,34,31,30,25,26], \
            [36,33,32,29,28,27]], 'i4') 
    a_equal(cube.sfc, expect)

    cube = CubePartition(ne=2*5, nproc=1)
    expect = np.array([ \
            [ 1, 2,31,32,33,36,37,40,41,42], \
            [ 4, 3,30,29,34,35,38,39,44,43], \
            [ 5, 6,27,28,23,22,51,50,45,46], \
            [ 8, 7,26,25,24,21,52,49,48,47], \
            [ 9,12,13,16,17,20,53,56,57,58], \
            [10,11,14,15,18,19,54,55,60,59], \
            [95,94,91,90,79,78,75,74,61,62], \
            [96,93,92,89,80,77,76,73,64,63], \
            [97,98,87,88,81,82,71,72,65,66], \
           [100,99,86,85,84,83,70,69,68,67]], 'i4') 
    a_equal(cube.sfc, expect)




def test_elem_gseq():
    '''
    elem_gseq: ne=3, nproc=1
    '''
    from cube_partition import CubePartition

    ne = 3
    expect_p1 = np.array( \
            [[ 1, 4, 5], \
             [ 2, 3, 6], \
             [ 9, 8, 7]])

    expect_p2 = np.array( \
            [[10,11,18], \
             [13,12,17], \
             [14,15,16]])

    expect_p3 = np.array( \
            [[50,51,52], \
             [49,48,53], \
             [46,47,54]])

    expect_p4 = np.array( \
            [[34,33,32], \
             [35,30,31], \
             [36,29,28]])

    expect_p5 = np.array( \
            [[45,38,37], \
             [44,39,40], \
             [43,42,41]])

    expect_p6 = np.array( \
            [[27,26,25], \
             [20,21,24], \
             [19,22,23]])


    #np.set_printoptions(threshold=np.nan)
    cube = CubePartition(ne, nproc=1)
    a_equal(cube.elem_gseq[0,:,:], expect_p1)
    a_equal(cube.elem_gseq[1,:,:], expect_p2)
    a_equal(cube.elem_gseq[2,:,:], expect_p3)
    a_equal(cube.elem_gseq[3,:,:], expect_p4)
    a_equal(cube.elem_gseq[4,:,:], expect_p5)
    a_equal(cube.elem_gseq[5,:,:], expect_p6)




def test_elem_proc_34():
    '''
    elem_proc: ne=3, nproc=4
    '''
    from cube_partition import CubePartition

    expect_p1 = np.array( \
            [[ 0, 0, 0], \
             [ 0, 0, 0], \
             [ 0, 0, 0]])

    expect_p2 = np.array( \
            [[ 0, 0, 1], \
             [ 0, 0, 1], \
             [ 0, 1, 1]])

    expect_p3 = np.array( \
            [[ 3, 3, 3], \
             [ 3, 3, 3], \
             [ 3, 3, 3]])

    expect_p4 = np.array( \
            [[ 2, 2, 2], \
             [ 2, 2, 2], \
             [ 2, 2, 1]])

    expect_p5 = np.array( \
            [[ 3, 2, 2], \
             [ 3, 2, 2], \
             [ 3, 3, 2]])

    expect_p6 = np.array( \
            [[ 1, 1, 1], \
             [ 1, 1, 1], \
             [ 1, 1, 1]])

    cube = CubePartition(ne=3, nproc=4)
    a_equal(cube.nelems, [14, 14, 13, 13])
    a_equal(cube.elem_proc[0,:,:], expect_p1)
    a_equal(cube.elem_proc[1,:,:], expect_p2)
    a_equal(cube.elem_proc[2,:,:], expect_p3)
    a_equal(cube.elem_proc[3,:,:], expect_p4)
    a_equal(cube.elem_proc[4,:,:], expect_p5)
    a_equal(cube.elem_proc[5,:,:], expect_p6)




def test_set_elem_proc_37():
    '''
    elem_proc: ne=3, nproc=7
    '''
    from cube_partition import CubePartition

    expect_p1 = np.array( \
            [[ 0, 0, 0], \
             [ 0, 0, 0], \
             [ 1, 0, 0]])

    expect_p2 = np.array( \
            [[ 1, 1, 2], \
             [ 1, 1, 2], \
             [ 1, 1, 1]])

    expect_p3 = np.array( \
            [[ 6, 6, 6], \
             [ 6, 6, 6], \
             [ 5, 5, 6]])

    expect_p4 = np.array( \
            [[ 4, 4, 3], \
             [ 4, 3, 3], \
             [ 4, 3, 3]])

    expect_p5 = np.array( \
            [[ 5, 4, 4], \
             [ 5, 4, 4], \
             [ 5, 5, 5]])

    expect_p6 = np.array( \
            [[ 3, 3, 3], \
             [ 2, 2, 2], \
             [ 2, 2, 2]])

    cube = CubePartition(ne=3, nproc=7)
    a_equal(cube.nelems, [8, 8, 8, 8, 8, 7, 7])
    a_equal(cube.elem_proc[0,:,:], expect_p1)
    a_equal(cube.elem_proc[1,:,:], expect_p2)
    a_equal(cube.elem_proc[2,:,:], expect_p3)
    a_equal(cube.elem_proc[3,:,:], expect_p4)
    a_equal(cube.elem_proc[4,:,:], expect_p5)
    a_equal(cube.elem_proc[5,:,:], expect_p6)
