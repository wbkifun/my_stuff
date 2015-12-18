#------------------------------------------------------------------------------
# filename  : test_sem_derivatives_plane.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.7     revise
#
#
# description: 
#   Derivatives using the Spectral Element Method on the plane (strong form)
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

from util.quadrature import legendre, deriv_legendre, gausslobatto




class SpectralElementPlane(object):
    def __init__(self, ngq):
        self.ngq = ngq

        self.dvv = np.zeros((ngq,ngq), 'f8')    # derivative matrix
        self.init_dvv()



    def init_dvv(self):
        ngq = self.ngq
        p_order = ngq-1
        gq_pts, gq_wts = gausslobatto(p_order) 
        dvv = self.dvv

        for i in xrange(ngq):
            for j in xrange(ngq):
                if i != j:
                    dvv[i,j] = 1/(gq_pts[i] - gq_pts[j]) * \
                            ( legendre(p_order,gq_pts[i]) / legendre(p_order,gq_pts[j]) )

                else:
                    if i == 0:
                        dvv[i,j] = - p_order*(p_order+1)/4

                    elif i == p_order:
                        dvv[i,j] = p_order*(p_order+1)/4

                    elif 0 < i < p_order:
                        dvv[i,j] = 0



    def gradient(self, scalar, ret):
        ngq = self.ngq
        dvvT = self.dvv.T
        assert scalar.shape == (ngq,ngq)    # scalar field
        assert ret.shape == (ngq,ngq,2)     # vector field

        scalar_flat = scalar.ravel()
        ret_flat = ret.ravel()
        dvvT_flat = dvvT.ravel()
        dvv_flat = self.dvv.ravel()

        for idx in xrange(ngq*ngq):
            i = idx//ngq 
            j = idx%ngq 

            tmpx, tmpy = 0, 0
            for k in xrange(ngq):
                tmpx += dvv_flat[i*ngq+k] * scalar_flat[k*ngq+j]
                tmpy += scalar_flat[i*ngq+k] * dvv_flat[j*ngq+k]

            ret_flat[(i*ngq+j)*2+0] = tmpx
            ret_flat[(i*ngq+j)*2+1] = tmpy


        '''
        for i in xrange(ngq):
            for j in xrange(ngq):
                tmpx, tmpy = 0, 0

                for k in xrange(ngq):
                    tmpx += dvvT[k,i] * scalar[k,j]
                    tmpy += scalar[i,k] * dvvT[k,j]

                ret[i,j,0] = tmpx
                ret[i,j,1] = tmpy

        return ret
        '''



    def divergence(self, vector, ret):
        ngq = self.ngq
        dvvT = self.dvv.T
        assert vector.shape == (ngq,ngq,2)  # vector field
        assert ret.shape == (ngq,ngq)       # scalar field

        for i in xrange(ngq):
            for j in xrange(ngq):
                tmpx, tmpy = 0, 0

                for k in xrange(ngq):
                    tmpx += dvvT[k,i] * vector[k,j,0]
                    tmpy += vector[i,k,1] * dvvT[k,j]

                ret[i,j] = tmpx + tmpy

        return ret



    def vorticity(self, vector, ret):
        ngq = self.ngq
        dvvT = self.dvv.T
        assert vector.shape == (ngq,ngq,2)  # vector field
        assert ret.shape == (ngq,ngq)       # scalar field

        for i in xrange(ngq):
            for j in xrange(ngq):
                tmpx, tmpy = 0, 0

                for k in xrange(ngq):
                    tmpx += dvvT[k,i] * vector[k,j,1]
                    tmpy += vector[i,k,0] * dvvT[k,j]

                ret[i,j] = tmpx - tmpy

        return ret



    def laplacian(self, scalar, ret):
        ngq = self.ngq
        assert scalar.shape == (ngq,ngq)    # scalar field
        assert ret.shape == (ngq,ngq)       # scalar field

        ret_tmp = np.zeros((ngq,ngq,2), 'f8')
        self.gradient(scalar, ret_tmp)
        return self.divergence(ret_tmp, ret)




def test_gradient():
    '''
    SEM on the plane: gradient(), f=x^2+2*y^2 on [-1,1]^2
    '''
    ngq = 4

    gq_pts, gq_wts = gausslobatto(ngq-1) 
    x = gq_pts[:,np.newaxis]
    y = gq_pts[np.newaxis,:]

    scalar = np.zeros((ngq,ngq), 'f8')
    ret = np.zeros((ngq,ngq,2), 'f8')

    sep = SpectralElementPlane(ngq)
    scalar[:] = x*x + 2*y*y

    sep.gradient(scalar, ret)
    aa_equal(ret[:,:,0], 2*x+0*y, 15)
    aa_equal(ret[:,:,1], 0*x+4*y, 15)




def test_divergence():
    '''
    SEM on the plane: divergence(), F=(x^3,2*y^2) on [-1,1]^2
    '''
    ngq = 4

    gq_pts, gq_wts = gausslobatto(ngq-1) 
    x = gq_pts[:,np.newaxis]
    y = gq_pts[np.newaxis,:]

    vector = np.zeros((ngq,ngq,2), 'f8')
    ret = np.zeros((ngq,ngq), 'f8')

    sep = SpectralElementPlane(ngq)
    vector[:,:,0] = x*x*x
    vector[:,:,1] = 2*y*y

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

    vector = np.zeros((ngq,ngq,2), 'f8')
    ret = np.zeros((ngq,ngq), 'f8')

    sep = SpectralElementPlane(ngq)
    vector[:,:,0] = 2*y*y
    vector[:,:,1] = x*x*x

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
    ngq = 4

    sep = SpectralElementPlane(ngq)

    scalar = np.random.rand(ngq,ngq)
    ret1 = np.zeros((ngq,ngq,2), 'f8')
    ret2 = np.zeros((ngq,ngq), 'f8')

    sep.gradient(scalar, ret1)
    sep.vorticity(ret1, ret2)
    aa_equal(ret2, np.zeros((ngq,ngq), 'f8'), 14)




if __name__ == '__main__':
    ngq = 4
    gq_pts, gq_wts = gausslobatto(ngq-1) 
    sep = SpectralElementPlane(ngq)
    print 'derivative matrix (dvvT)\n', sep.dvv.T
