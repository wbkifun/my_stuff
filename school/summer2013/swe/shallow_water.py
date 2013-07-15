from __future__ import division
import numpy
import os

from pykgm.share.physical_constant import reverse_earth_radius as rrearth
from pykgm.share.physical_constant import gravitational_const as gconst
from pykgm.share.source_module_f90 import get_module_f90

from interact_between_elems import InteractBetweenElems




class CubedSphereGrid(object):
    def __init__(self, ne, ngq):
        basepath = os.path.expanduser('~/.pykgm/cs_grid/')
        dpath = basepath + 'ne%d_ngq%d/'%(ne,ngq)

        self.ne = ne
        self.ngq = ngq

        self.ab = numpy.load(dpath+'alpha_beta.npy')
        self.lonlat = numpy.load(dpath+'lonlat.npy')
        self.mvp_coord = numpy.load(dpath+'mvp_coord.npy')
        self.mvp_num = numpy.load(dpath+'mvp_num.npy')
        self.is_uvp = numpy.load(dpath+'is_uvp.npy')

        self.uvp_size = self.is_uvp.sum()




class State(object):
    def __init__(self, csgrid):
        self.csgrid = csgrid

        ne = csgrid.ne
        ngq = csgrid.ngq


        self.nelem = nelem = ne*ne*6
        self.shape = shape = (ngq,ngq,1,nelem)

        # velocity
        self.u = numpy.zeros(shape, 'f8', order='F')
        self.v = numpy.zeros(shape, 'f8', order='F')

        # depth of fluid
        self.h = numpy.zeros(shape, 'f8', order='F')

        # Coriolis parameter
        self.fcor = numpy.zeros(shape, 'f8', order='F')



    def save_ascii(self, fp, tstep):
        ne = self.csgrid.ne
        ngq = self.csgrid.ngq
        nelem = self.nelem
        uvp_size = self.csgrid.uvp_size
        lonlat = self.csgrid.lonlat
        is_uvp = self.csgrid.is_uvp
        u, v, h = self.u, self.v, self.h

        if tstep == 1:
            fp.write('lon\tlat\tu\tv\th\n') 
            fp.write('%d\n' % uvp_size)

        for gj in xrange(ngq):
            for gi in xrange(ngq):
                for ie in xrange(nelem):
                    ei = ie%ne
                    ej = (ie%(ne*ne))/ne
                    face = ie/(ne*ne)
                    coord = (gi,gj,0,ie)

                    if is_uvp[gi,gj,ei,ej,face]:
                        lon, lat = lonlat[:,gi,gj,ei,ej,face]
                        
                        fp.write('%f\t%f\t%f\t%f\t%f\n' % 
                                (lon, lat, u[coord], v[coord], h[coord]))




class InsideElems(object):
    def __init__(self, csgrid, state, interact):
        self.csgrid = csgrid
        self.state = state
        self.interact = interact

        self.ne = ne = csgrid.ne
        self.ngq = ngq = csgrid.ngq
        self.nelem = nelem = state.nelem


        # transform matrix
        self.A = numpy.zeros((2,2,ngll,ngll,nelem), 'f8', order='F')
        self.AI = numpy.zeros((2,2,ngll,ngll,nelem), 'f8', order='F')
        self.J = numpy.zeros((ngll,ngll,nelem), 'f8', order='F')
        self.A[:] = ncf.variables['A'][:]
        self.AI[:] = ncf.variables['AI'][:]
        self.J[:] = ncf.variables['J'][:]


        # derivative matrix
        self.dvv = numpy.zeros((ngll,ngll), 'f8', order='F')
        self.dvvT = numpy.zeros((ngll,ngll), 'f8', order='F')
        self.dvv[:] = ncf.variables['dvv'][:]
        self.dvvT[:] = self.dvv.T


        # prepare interaction
        self.ret_u = numpy.zeros(state.shape, 'f8', order='F')
        self.ret_v = numpy.zeros(state.shape, 'f8', order='F')
        self.ret_h = numpy.zeros(state.shape, 'f8', order='F')
        interact.set_variables([self.ret_u, self.ret_v, self.ret_h])


        # prepare the compute_rhs
        mod = get_module_f90( open('sem_cores.f90').read() )
        self.compute_rhs_f90 = mod.compute_rhs_f90
        #from pykgm.trial.spectral_element.advection_2d_sphere.sem_cores \
        #        import compute_rhs_f90
        #self.compute_rhs_f90 = compute_rhs_f90



    def compute_rhs(self, tn, u, v, h, ku, kv, kh):
        dvvT, J, A, AI = self.dvvT, self.J, self.A, self.AI
        state = self.state
        ret_u, ret_v, ret_h = self.ret_u, self.ret_v, self.ret_h
        interact = self.interact

        self.compute_rhs_f90(rrearth, gconst,
                             dvvT, J, A, AI,
                             state.fcor, u, v, h,
                             ret_u, ret_v, ret_h)

        #interact.start_buf_exchange()
        #interact.wait_buf_exchange()
        #interact.interact_between_elems_buf()
        interact.interact_between_elems_inner()

        ku[:], kv[:], kh[:] = ret_u, ret_v, ret_h




class RungeKutta(object):
    def __init__(self, dt, compute_rhs):
        self.dt = dt
        self.tn = dt
        self.compute_rhs = compute_rhs



    def allocate(self, shape):
        self.ku = [numpy.zeros(shape, 'f8', order='F') for i in xrange(4)]
        self.kv = [numpy.zeros(shape, 'f8', order='F') for i in xrange(4)]
        self.kh = [numpy.zeros(shape, 'f8', order='F') for i in xrange(4)]



    def update_rk4(self, u, v, h):
        dt = self.dt
        hdt = self.dt*0.5
        tn = self.tn
        ku, kv, kh = self.ku, self.kv, self.kh
        compute_rhs = self.compute_rhs

        compute_rhs(tn,     u,           v,           h,           ku[0], kv[0], kh[0]) 
        compute_rhs(tn+hdt, u+hdt*ku[0], v+hdt*kv[0], h+hdt*kh[0], ku[1], kv[1], kh[1]) 
        compute_rhs(tn+hdt, u+hdt*ku[1], v+hdt*kv[1], h+hdt*kh[1], ku[2], kv[2], kh[2]) 
        compute_rhs(tn+dt,  u+dt*ku[2],  v+dt*kv[2],  h+dt*kh[2],  ku[3], kv[3], kh[3])

        u[:] += (dt/6)*(ku[0] + 2*ku[1] + 2*ku[2] + ku[3])
        v[:] += (dt/6)*(kv[0] + 2*kv[1] + 2*kv[2] + kv[3])
        h[:] += (dt/6)*(kh[0] + 2*kh[1] + 2*kh[2] + kh[3])
        tn += dt


'''
    def update_rk3(self, psi):
        dt = self.dt
        k1, k2 = self.k1, self.k2
        compute_rhs = self.compute_rhs

        k1[:] = psi + dt*compute_rhs(psi) 
        k2[:] = (3/4)*psi + (1/4)*k1 + (1/4)*dt*compute_rhs(k1) 
        psi[:] = (1/3)*psi + (2/3)*k2 + (2/3)*dt*compute_rhs(k2) 
'''



if __name__ == '__main__':
    from numpy import pi, sin, cos

    from pykgm.share.physical_constant import rotating_velocity as omega
    from pykgm.share.physical_constant import earth_radius as rearth
    from pykgm.share.physical_constant import gravitational_const as gconst


    ne, ngq = 10, 4
    csgrid = CubedSphereGrid(ne, ngq)
    state = State(csgrid)

    #----------------------------------------------
    # initialize the states
    #----------------------------------------------
    u0 = 2*pi*rearth/(12*24*3600)  # 12 days
    h0 = 2.94*1e4/gconst

    for face in xrange(6):
        for ej in xrange(ne):
            for ei in xrange(ne):
                ie = ei + ej*ne + face*ne*ne

                lats = csgrid.lonlat[1,:,:,ei,ej,face]
                state.fcor[:,:,0,ie] = 2*omega*sin(lats)
                state.u[:,:,0,ie] = u0*cos(lats)
                state.h[:,:,0,ie] = h0 - (2*rearth*omega + u0)*u0*sin(lats)**2 / (2*gconst)

    f = open('uvh2.ascii', 'w')
    state.save_ascii(f, tstep=1)
    f.close()
