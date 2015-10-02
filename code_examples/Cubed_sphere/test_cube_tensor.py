#------------------------------------------------------------------------------
# filename  : test_cube_tensor.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.25     start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
from math import fsum

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

from pkg.util.compare_float import feq
from cube_mpi import CubeGridMPI
from cube_tensor import CubeTensor




def test_jacobian_area_30_4():
    '''
    CubeTensor: Jacobian area test (ne=30, ngq=4, nproc=1)
    '''
    ne, ngq = 30, 4
    nproc, myrank = 1, 0

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubetensor = CubeTensor(cubegrid)

    local_ep_size = cubegrid.local_ep_size
    local_gq_indices = cubegrid.local_gq_indices
    J = cubetensor.J
    gq_wts = cubetensor.gq_wts

    areas = np.zeros(local_ep_size)
    for seq in xrange(local_ep_size):
        panel, ei, ej, gi, gj = local_gq_indices[seq]
        areas[seq] = J[seq]*gq_wts[gi-1]*gq_wts[gj-1]

    #aa_equal(fsum(areas), 4*np.pi, 11)
    #aa_equal(fsum(areas)/100, 4*np.pi/100, 13)  # /100 for accuracy comparison
    ok_( feq(fsum(areas), 4*np.pi, 13) )




def test_jacobian_area_30_4_2():
    '''
    CubeTensor: Jacobian area test (ne=30, ngq=4, nproc=2)
    '''
    ne, ngq = 30, 4
    nproc = 2

    # Rank 0
    cubegrid0 = CubeGridMPI(ne, ngq, nproc, myrank=0)
    cubetensor0 = CubeTensor(cubegrid0)

    local_ep_size0 = cubegrid0.local_ep_size
    local_gq_indices0 = cubegrid0.local_gq_indices
    J0 = cubetensor0.J
    gq_wts0 = cubetensor0.gq_wts

    areas0 = np.zeros(local_ep_size0)
    for seq in xrange(local_ep_size0):
        panel, ei, ej, gi, gj = local_gq_indices0[seq]
        areas0[seq] = J0[seq]*gq_wts0[gi-1]*gq_wts0[gj-1]


    # Rank 1
    cubegrid1 = CubeGridMPI(ne, ngq, nproc, myrank=1)
    cubetensor1 = CubeTensor(cubegrid1)

    local_ep_size1 = cubegrid1.local_ep_size
    local_gq_indices1 = cubegrid1.local_gq_indices
    J1 = cubetensor1.J
    gq_wts1 = cubetensor1.gq_wts

    areas1 = np.zeros(local_ep_size1)
    for seq in xrange(local_ep_size1):
        panel, ei, ej, gi, gj = local_gq_indices1[seq]
        areas1[seq] = J1[seq]*gq_wts1[gi-1]*gq_wts1[gj-1]


    ok_( feq(fsum(areas0)+fsum(areas1), 4*np.pi, 13) )
