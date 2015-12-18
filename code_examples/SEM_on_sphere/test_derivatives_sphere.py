from __future__ import division
import numpy
from numpy import pi, sin, cos, tan, abs, sqrt
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal


from pykgm.trial.parallel_sfc.io_preprocess import load_preprocess_netcdf
from cubed_sphere.cube_tensor import CubeTensor
from cubed_sphere.cube_mpi import CubeGridMPI, CubeMPI



class SpectralElementSphere(object):
    def __init__(self, ncf, ie):
        self.N = N = ncf.N
        self.ngll = ngll = ncf.ngll
        self.nelem = nelem = ncf.nelem
        self.ie = ie


        # transform matrix
        self.A = numpy.zeros((2,2,ngll,ngll,nelem), 'f8')
        self.AI = numpy.zeros((2,2,ngll,ngll,nelem), 'f8', order='F')
        self.J = numpy.zeros((ngll,ngll,nelem), 'f8')
        self.A[:] = ncf.variables['A'][:]
        self.AI[:] = ncf.variables['AI'][:]
        self.J[:] = ncf.variables['J'][:]

        #print 'J', self.J[1,1,0]


        # derivative matrix
        self.dvv = numpy.zeros((ngll,ngll), 'f8')
        self.dvvT = numpy.zeros((ngll,ngll), 'f8')
        self.dvv[:] = ncf.variables['dvv'][:]
        self.dvvT[:] = self.dvv.T


        # compare
        cubegrid = CubeGridMPI(N, ngll, nproc=1, myrank=0)
        cubetensor = CubeTensor(cubegrid)

        lonlat_coord = ncf.variables['lonlat_coord'][:]
        lons = lonlat_coord[0,:,:,:]
        lats = lonlat_coord[1,:,:,:]

        a_equal(lats.ravel(), cubegrid.local_latlons[:,0])
        a_equal(lons.ravel(), cubegrid.local_latlons[:,1])

        '''
        AI = ArrayAs(platform, cubetensor.AI, 'AI')     # local_ep_size*2*2
        J = ArrayAs(platform, cubetensor.J, 'J')        # local_ep_size
        dvv = ArrayAs(platform, cubetensor.dvv, 'dvvT') # ngq*ngq
        '''
        aa_equal(self.dvv.ravel(), cubetensor.dvv, 15)
        #aa_equal(self.J.ravel(), cubetensor.J, 15)
        aa_equal(self.AI.ravel(), cubetensor.AI, 15)



    def gradient(self, scalar):
        ngll = self.ngll
        dvvT = self.dvvT
        AI = self.AI[:,:,:,:,self.ie-1]
        assert scalar.shape == (ngll,ngll)

        ret = numpy.zeros((2,ngll,ngll), 'f8')

        for gj in xrange(ngll):
            for gi in xrange(ngll):
                tmpx, tmpy = 0, 0

                for k in xrange(ngll):
                    tmpx += dvvT[k,gi] * scalar[k,gj]
                    tmpy += scalar[gi,k] * dvvT[k,gj]

                # co -> latlon (AIT)
                ret[0,gi,gj] = AI[0,0,gi,gj]*tmpx + AI[1,0,gi,gj]*tmpy
                ret[1,gi,gj] = AI[0,1,gi,gj]*tmpx + AI[1,1,gi,gj]*tmpy

        return ret



    def divergence(self, vector):
        ngll = self.ngll
        dvvT = self.dvvT
        AI = self.AI[:,:,:,:,self.ie-1]
        J = self.J[:,:,self.ie-1]
        assert vector.shape == (2,ngll,ngll)

        jcontra = numpy.zeros((2,ngll,ngll), 'f8', order='F')
        ret = numpy.zeros((ngll,ngll), 'f8', order='F')

        # latlon -> contra (AI)
        for gj in xrange(ngll):
            for gi in xrange(ngll):
                jcontra[0,gi,gj] = (AI[0,0,gi,gj]*vector[0,gi,gj] + \
                                    AI[0,1,gi,gj]*vector[1,gi,gj]) * J[gi,gj]
                jcontra[1,gi,gj] = (AI[1,0,gi,gj]*vector[0,gi,gj] + \
                                    AI[1,1,gi,gj]*vector[1,gi,gj]) * J[gi,gj]

        for gj in xrange(ngll):
            for gi in xrange(ngll):
                tmpx, tmpy = 0, 0

                for k in xrange(ngll):
                    tmpx += dvvT[k,gi] * jcontra[0,k,gj]
                    tmpy += jcontra[1,gi,k] * dvvT[k,gj]

                ret[gi,gj] = (tmpx + tmpy)/J[gi,gj]

        return ret



    def vorticity(self, vector):
        ngll = self.ngll
        dvvT = self.dvvT
        A = self.A[:,:,:,:,self.ie-1]
        J = self.J[:,:,self.ie-1]
        assert vector.shape == (2,ngll,ngll)

        co = numpy.zeros((2,ngll,ngll), 'f8', order='F')
        ret = numpy.zeros((ngll,ngll), 'f8', order='F')

        # latlon -> co (AT)
        for gj in xrange(ngll):
            for gi in xrange(ngll):
                co[0,gi,gj] = A[0,0,gi,gj]*vector[0,gi,gj] + \
                              A[1,0,gi,gj]*vector[1,gi,gj]
                co[1,gi,gj] = A[0,1,gi,gj]*vector[0,gi,gj] + \
                              A[1,1,gi,gj]*vector[1,gi,gj]

        for gj in xrange(ngll):
            for gi in xrange(ngll):
                tmpx, tmpy = 0, 0

                for k in xrange(ngll):
                    tmpx += dvvT[k,gi] * co[1,k,gj]
                    tmpy += co[0,gi,k] * dvvT[k,gj]

                ret[gi,gj] = (tmpx - tmpy)/J[gi,gj]

        return ret



    def laplacian(self, scalar):
        ngll = self.ngll
        assert scalar.shape == (ngll,ngll)

        return self.divergence( self.gradient(scalar) )




def get_rms_max_min(nu, ana=None):
    if ana != None:
        rel = (ana - nu)/ana
        rel_abs = abs(rel)
        rms = sqrt( numpy.average(rel**2) )
    else:
        rel_abs = abs(nu)
        rms = sqrt( numpy.average(nu**2) )

    return rms, rel_abs.max(), rel_abs.min()




if __name__ == '__main__':
    print '-'*80
    print 'Test the derivative operators on the sphere'
    print 'using the spectral element method with strong form'
    print '-'*80


    #---------------------------------------------------
    # setup
    #---------------------------------------------------
    N = 30          # elements / axis
    ngll = 4        # GLL points / axis / element
    cfl = 0.2       # Courant-Friedrichs-Lewy condition 
    nproc = 1
    rank = 1
    ie = 1

    # load the preprocessed netcdf file
    ncf = load_preprocess_netcdf(N, ngll, nproc, rank)
    nelem = ncf.nelem    # total elements
    lonlat_coord = ncf.variables['lonlat_coord'][:]
    lons = lonlat_coord[0,:,:,ie-1]
    lats = lonlat_coord[1,:,:,ie-1]

    # test fields
    scalar_field = numpy.zeros((ngll,ngll), 'f8', order='F')
    vector_field = numpy.zeros((2,ngll,ngll), 'f8', order='F')

    # spectral element method
    ses = SpectralElementSphere(ncf, ie)


    """
    #---------------------------------------------------
    # dvv matrix
    #---------------------------------------------------
    print 'derivative matrix (dvvT)\n\n', ses.dvvT
    print '-'*80


    #---------------------------------------------------
    # Gradient
    #---------------------------------------------------
    print 'gradient()\n'

    #scalar_field[:] = lons + 2*lats
    #print 'input: scalar field:  f = lon + 2*lat\n'
    scalar_field[:] = sin(lons)*cos(lats)
    print 'input: scalar field:  f = sin(lons)*cos(lats)'

    ret = ses.gradient(scalar_field)
    #ana_x = 1/cos(lats)
    #ana_y = 2
    ana_x = cos(lons)
    ana_y = -sin(lons)*sin(lats)

    #print 'diff_x:\n', ana_x - ret[0,:,:]
    #print 'diff_y:\n', ana_y - ret[1,:,:] 

    print 'lon) rms: %g, rel_max: %g, rel_min: %g' % get_rms_max_min(ret[0,:,:], ana_x)
    print 'lat) rms: %g, rel_max: %g, rel_min: %g' % get_rms_max_min(ret[1,:,:], ana_y)
    print '-'*80


    #---------------------------------------------------
    # Divergence
    #---------------------------------------------------
    print 'divergence()\n'

    vector_field[0,:,:] = lons
    vector_field[1,:,:] = 2*lats
    print 'input: vector field:  F = (lon,2*lat)'

    ret = ses.divergence(vector_field)
    ana = (1 + 2*cos(lats) - 2*lats*sin(lats))/cos(lats)

    #print 'diff:\n', ana - ret 
    print 'rms: %g, rel_max: %g, rel_min: %g' % get_rms_max_min(ret, ana)
    print '-'*80

    
    #---------------------------------------------------
    # Vorticity
    #---------------------------------------------------
    print 'vorticity()\n'

    vector_field[0,:,:] = lons
    vector_field[1,:,:] = 2*lats
    print 'input: vector field:  F = (lon,2*lat)'

    ret = ses.vorticity(vector_field)
    ana = lons*tan(lats)

    #print 'diff:\n', ana - ret 
    print 'rms: %g, rel_max: %g, rel_min: %g' % get_rms_max_min(ret, ana)
    print '-'*80


    #---------------------------------------------------
    # Laplacian
    #---------------------------------------------------
    print 'laplacian()\n'

    scalar_field[:,:] = lons + 2*lats
    print 'input: scalar field:  f = lon^2 + 2*lat'

    ret = ses.laplacian(scalar_field)
    ana = -2*tan(lats)

    #print 'diff:\n', ana - ret 
    print 'rms: %g, rel_max: %g, rel_min: %g' % get_rms_max_min(ret, ana)
    print '-'*80
    

    #---------------------------------------------------
    # Zero identity
    #---------------------------------------------------
    print 'zero identity: vorticity( gradient() )\n'

    scalar_field[:] = numpy.random.rand(ngll,ngll)
    print 'input: scalar field (random)'

    ret = ses.vorticity( ses.gradient(scalar_field) )

    #print 'output:\n', ret
    print 'rms: %g, rel_max: %g, rel_min: %g' % get_rms_max_min(ret)
    print '-'*80
    """
