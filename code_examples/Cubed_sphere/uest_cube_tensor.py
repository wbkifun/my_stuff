#------------------------------------------------------------------------------
# filename  : test_cube_tensor.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.25     start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
from numpy import pi, sin, cos, tan, sqrt
from math import fsum

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

from pkg.util.compare_float import feq
from cube_mpi import CubeGridMPI
from cube_tensor import CubeTensor




def test_jacobian_area_30():
    '''
    CubeTensor: Jacobian area test (ne=30, ngq=4)
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




def test_jacobian_area_30_2():
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




def test_transform_matrix_gradient():
    '''
    CubeTensor: Transform matrix, gradient test (ne=30, ngq=4)
    '''
    ne, ngq = 30, 4
    nproc, myrank = 1, 0

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubetensor = CubeTensor(cubegrid)

    # In an element
    dvvT = cubetensor.dvvT
    AI = cubetensor.AI[:ngq*ngq*2*2]
    lats = cubegrid.local_latlons[:ngq*ngq,0]
    lons = cubegrid.local_latlons[:ngq*ngq,1]
    #AI = cubetensor.AI[-ngq*ngq*2*2:]
    #lats = cubegrid.local_latlons[-ngq*ngq:,0]
    #lons = cubegrid.local_latlons[-ngq*ngq:,1]

    #scalar = sin(lons)*cos(lats)
    scalar = -0.5*sqrt(1.5/pi)*sin(lats)*cos(lons)   # Y11
    ret_lat = np.zeros(ngq*ngq, 'f8')
    ret_lon = np.zeros(ngq*ngq, 'f8')

    for idx in xrange(ngq*ngq):
        i = idx%ngq     # inmost order
        j = idx//ngq    # outmost order

        tmpx, tmpy = 0, 0
        for k in xrange(ngq):
            tmpx += dvvT[i*ngq+k]*scalar[j*ngq+k]
            tmpy += scalar[k*ngq+i]*dvvT[j*ngq+k]

        # covariant -> latlon (AIT)
        ret_lon[idx] = AI[idx*4+0]*tmpx + AI[idx*4+2]*tmpy
        ret_lat[idx] = AI[idx*4+1]*tmpx + AI[idx*4+3]*tmpy

    #aa_equal(ret_lon, cos(lons), 4)
    #aa_equal(ret_lat, -sin(lons)*sin(lats), 6)
    aa_equal(ret_lon, 0.5*sqrt(1.5/pi)*tan(lats)*sin(lons), 5)
    aa_equal(ret_lat, -0.5*sqrt(1.5/pi)*cos(lats)*cos(lons), 5)




def test_transform_matrix_inner():
    '''
    CubeTensor: Transform matrix, inner product test (ne=30, ngq=4)
    '''
    ne, ngq = 30, 4
    nproc, myrank = 1, 0

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubetensor = CubeTensor(cubegrid)

    for idx in xrange(cubegrid.local_ep_size):
        AI = cubetensor.AI[idx*4:idx*4+4].reshape(2,2)
        A = np.linalg.inv(AI)
        g = np.dot(A.T,A)    # metric tensor
        
        # inner product in the latlon coordinates
        v1_lon, v1_lat = np.random.rand(2)
        v2_lon, v2_lat = np.random.rand(2)
        ip_ll = v1_lon*v2_lon + v1_lat*v2_lat

        # inner product in the cubed-sphere coordinates
        # latlon -> contravariant
        v1_0 = AI[0,0]*v1_lon + AI[0,1]*v1_lat
        v1_1 = AI[1,0]*v1_lon + AI[1,1]*v1_lat
        v2_0 = AI[0,0]*v2_lon + AI[0,1]*v2_lat
        v2_1 = AI[1,0]*v2_lon + AI[1,1]*v2_lat
        ip_xy = g[0,0]*v1_0*v2_0 + g[0,1]*v1_0*v2_1 \
              + g[1,0]*v1_1*v2_0 + g[1,1]*v1_1*v2_1

        aa_equal(ip_ll, ip_xy, 15)




def test_compare_homme():
    '''
    CubeTensor: Compare the transform matrix with HOMME (ne=30, ngq=4)
    '''
    ne, ngq = 3, 4
    nproc, myrank = 1, 0

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubetensor = CubeTensor(cubegrid)
    cubetensor_homme = CubeTensor(cubegrid, homme_style=True)

    aa_equal(cubetensor.AI, cubetensor_homme.AI)
    aa_equal(cubetensor.J, cubetensor_homme.J)
