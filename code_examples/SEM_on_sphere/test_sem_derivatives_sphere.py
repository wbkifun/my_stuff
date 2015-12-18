#------------------------------------------------------------------------------
# filename  : test_sem_derivatives_sphere.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.7     revise
#
#
# description: 
#   Derivatives using the Spectral Element Method on the sphere (strong form)
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy import pi, sin, cos, tan, sqrt
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

from multi_platform.machine_platform import MachinePlatform
from multi_platform.array_variable import Array, ArrayAs
from cubed_sphere.cube_tensor import CubeTensor
from cubed_sphere.cube_mpi import CubeGridMPI, CubeMPI




class SpectralElementSphere(object):
    def __init__(self, platform, cubegrid):
        self.platform = platform
        self.cubegrid = cubegrid

        ngq = cubegrid.ngq
        local_ep_size = cubegrid.local_ep_size


        #--------------------------------------------------------
        # Transform matrix, Jacobian, Derivative matrix
        #--------------------------------------------------------
        cubetensor = CubeTensor(cubegrid)

        AI = ArrayAs(platform, cubetensor.AI, 'AI')         #(local_ep_size*2*2)
        J = ArrayAs(platform, cubetensor.J, 'J')            #(local_ep_size)
        dvvT = ArrayAs(platform, cubetensor.dvvT, 'dvvT')   #(ngq*ngq)


        #--------------------------------------------------------
        # Prepare the update_element function
        #--------------------------------------------------------
        src = open('./sem_derivatives.'+platform.code_type).read()
        pyf = open('./sem_derivatives.pyf').read()

        lib = platform.source_compile(src, pyf)
        self.gradient_core = platform.get_function(lib, 'gradient')
        #self.divergence_core = platform.get_function(lib, 'divergence')
        #self.vorticity_core = platform.get_function(lib, 'vorticity')
        #self.laplacian_core = platform.get_function(lib, 'laplacian')

        self.gradient_core.prepare('iioooOO', local_ep_size, ngq, dvvT, J, AI)



    def gradient(self, scalar, ret):
        local_ep_size = self.cubegrid.local_ep_size
        assert scalar.size == local_ep_size
        assert ret.size == local_ep_size*2

        self.gradient_core.prepared_call(scalar, ret)



    def divergence(self, vector, ret):
        ep_size = self.cubegrid.ep_size
        assert vector.shape == (ep_size,2)
        assert ret.shape == (ep_size,)



    def vorticity(self, vector, ret):
        ep_size = self.cubegrid.ep_size
        assert vector.shape == (ep_size,2)
        assert ret.shape == (ep_size,)



    def laplacian(self, scalar, ret):
        ep_size = self.cubegrid.ep_size
        assert scalar.size == ep_size
        assert ret.size == ep_size

        ret_tmp = np.zeros((ep_size,2), 'f8')
        self.gradient(scalar, ret_tmp)
        return self.divergence(ret_tmp, ret)



    def average_boundary(self, f):
        cubempi = CubeMPI(self.cubegrid, 'AVG')

        send_dsts = cubempi.send_dsts
        send_srcs = cubempi.send_srcs
        send_wgts = cubempi.send_wgts
        recv_dsts = cubempi.recv_dsts
        recv_srcs = cubempi.recv_srcs
        assert cubempi.local_src_size == len(send_dsts), 'local_src_size=%d, len(send_dsts)=%d'%(cubempi.local_src_size,len(send_dsts))

        recv_buf = np.zeros(cubempi.recv_buf_size, 'f8')

        for dst, src, wgt in zip(send_dsts, send_srcs, send_wgts):
            recv_buf[dst] += f[src]*wgt

        prev_dst = -1
        for dst, src in zip(recv_dsts, recv_srcs):
            if prev_dst != dst:
                f[dst] = 0
                prev_dst = dst
            f[dst] += recv_buf[src]
        



def test_gradient():
    '''
    SEM on the sphere: gradient(), Y11
    '''
    ne, ngq = 30, 4
    nproc, myrank = 1, 0

    # setup
    platform = MachinePlatform('CPU', 'f90')
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)

    local_ep_size = cubegrid.local_ep_size
    lats = cubegrid.local_latlons[:,0]
    lons = cubegrid.local_latlons[:,1]

    # test function, spherical harmonics Y25
    theta, phi = lons, pi/2-lats
    Y11 = -0.5*sqrt(1.5/pi)*sin(lats)*cos(lons)
    Y25 = (1/8)*sqrt(1155/(2*pi))*cos(2*theta)*sin(phi)**2*(3*cos(phi)**3-cos(phi))
    scalar = ArrayAs(platform, Y11)
    ret = Array(platform, local_ep_size*2, 'f8')

    sep = SpectralElementSphere(platform, cubegrid)
    sep.gradient(scalar, ret)
    #sep.average_boundary(ret.get()[::2])
    #sep.average_boundary(ret.get()[1::2])

    # verify
    ref_lon = 0.5*sqrt(1.5/pi)*tan(lats)*sin(lons)
    #ref_lon = -(1/4)*sqrt(1155/(2*pi))*sin(2*theta)*sin(phi)*(3*cos(phi)**3-cos(phi))

    for ie in xrange(local_ep_size//(ngq*ngq)):
        print cubegrid.local_gq_indices[ie*16]
        #idxs = [ie*16+k for k in [5,6,9,10]]
        #aa_equal(ret.get()[::2][idxs], ref_lon[idxs], 5)
        idxs = [ie*16+k for k in xrange(16)]
        aa_equal(ret.get()[::2][idxs], ref_lon[idxs], 5)

    #aa_equal(ret.get()[::2], 0.5*sqrt(1.5/pi)*tan(lats)*sin(lons), 3)  # lon
    #aa_equal(ret.get()[1::2], 0.5*sqrt(3/(2*pi))*cos(lats)*cos(lons), 5)  # lat



"""
def test_divergence():
    '''
    SEM on the plane: divergence(), F=(x^3,2*y^2) on [-1,1]^2
    '''
    ngq = 4

    gq_pts, gq_wts = gausslobatto(ngq-1) 
    x = gq_pts[:,np.newaxis]
    y = gq_pts[np.newaxis,:]

    vector = np.zeros((2,ngq,ngq), 'f8')
    ret = np.zeros((ngq,ngq), 'f8')

    sep = SpectralElementPlane(ngq)
    vector[0,:,:] = x*x*x
    vector[1,:,:] = 2*y*y

    sep.divergence(vector, ret)
    aa_equal(ret, 3*x*x+4*y, 15)




def test_vorticity():
    '''
    SEM on the plane: vorticity(), F=(2*y^2,x^3) on [-1,1]^2
    '''
    ngq = 4

    gq_pts, gq_wts = gausslobatto(ngq-1) 
    x = gq_pts[:,np.newaxis]
    y = gq_pts[np.newaxis,:]

    vector = np.zeros((2,ngq,ngq), 'f8')
    ret = np.zeros((ngq,ngq), 'f8')

    sep = SpectralElementPlane(ngq)
    vector[0,:,:] = 2*y*y
    vector[1,:,:] = x*x*x

    sep.vorticity(vector, ret)
    aa_equal(ret, 3*x*x-4*y, 15)




def test_laplacian():
    '''
    SEM on the plane: laplacian(), f=x^3+2*y^2 on [-1,1]^2
    '''
    ngq = 4

    gq_pts, gq_wts = gausslobatto(ngq-1) 
    x = gq_pts[:,np.newaxis]
    y = gq_pts[np.newaxis,:]

    scalar = np.zeros((ngq,ngq), 'f8')
    ret = np.zeros((ngq,ngq), 'f8')

    sep = SpectralElementPlane(ngq)
    scalar[:] = x*x*x + 2*y*y

    sep.laplacian(scalar, ret)
    aa_equal(ret, 6*x+0*y+4, 13)




def test_zero_identity():
    '''
    SEM on the plane: zero identity, vorticity(gradient())=0
    '''
    ne, ngq = 30, 4
    nproc, myrank = 1, 0

    platform = MachinePlatform('CPU', 'f90')
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)

    sep = SpectralElementSphere(platform, cubegrid)

    scalar = np.random.rand(ngq,ngq)
    ret1 = np.zeros((2,ngq,ngq), 'f8')
    ret2 = np.zeros((ngq,ngq), 'f8')

    sep.gradient(scalar, ret1)
    sep.vorticity(ret1, ret2)
    aa_equal(ret2, np.zeros((ngq,ngq), 'f8'), 14)
"""
