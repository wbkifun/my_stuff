from __future__ import division
import numpy
import sys
from numpy import pi, sin, cos, arccos, exp, abs

from runge_kutta import RungeKutta
from sem_cores import compute_rhs_f90
from interact_inner import interact_between_elems_inner_f90




class State(object):
    def __init__(self, ngll, nelem):
        ngll = ngll
        nelem = nelem

        self.psi = numpy.zeros((ngll,ngll,1,nelem), 'f8', order='F')
        self.velocity = numpy.zeros((2,ngll,ngll,1,nelem), 'f8', order='F')




class InteractBetweenElems(object):
    def __init__(self):
        # import the fortran subroutines
        self.interact_inner_f90 = interact_between_elems_inner_f90

        # fixed for N=120, ngll=4
        self.mvp_inner = numpy.load('./preprocess_files/mvp_inner.npy')



    def interact_between_elems_inner(self, var):
        mvp_inner = self.mvp_inner
        interact_inner_f90 = self.interact_inner_f90

        interact_inner_f90(mvp_inner, var)




class InsideElems(object):
    def __init__(self, N, ngll, nelem, state, interact):
        self.N = N 
        self.ngll = ngll 
        self.nelem = nelem 
        self.state = state
        self.interact = interact


        # transform matrix
        self.AI = numpy.zeros((2,2,ngll,ngll,nelem), 'f8', order='F')
        self.J = numpy.zeros((ngll,ngll,nelem), 'f8', order='F')
        self.AI[:] = numpy.load('./preprocess_files/AI.npy')
        self.J[:] = numpy.load('./preprocess_files/J.npy')


        # derivative matrix
        self.dvvT = numpy.zeros((ngll,ngll), 'f8', order='F')
        self.dvvT[:] = numpy.load('./preprocess_files/dvvT.npy')


        # prepare the compute_rhs
        self.compute_rhs_f90 = compute_rhs_f90



    def compute_rhs(self, psi, ret_psi):
        self.compute_rhs_f90(self.dvvT, self.J, self.AI, self.state.velocity, \
					         psi, ret_psi)
        self.interact.interact_between_elems_inner(ret_psi)




if __name__ == '__main__':
    #----------------------------------------------
    # setup
    #----------------------------------------------
    N = 120         # elements / axis
    ngll = 8        # GLL points / axis / element
    cfl = 0.1       # Courant-Friedrichs-Lewy condition 
    nproc = 1
    rank = 1


    nelem = N*N*6
    #min_dx = 0.0034069557991275559  # fixed for N=120, ngll=4
    min_dx = 0.00079122793323062573  # fixed for N=120, ngll=4
    lonlat_coord = numpy.load('./preprocess_files/lonlat_coord.npy')

    # state variables
    state = State(ngll, nelem)


    #----------------------------------------------
    # initialize the velocity vector and the scalar field
    #----------------------------------------------
    # initialize with the 2D gaussian
    lon0, lat0 = pi/2, -pi/5
    lons = lonlat_coord[0,:,:,:]
    lats = lonlat_coord[1,:,:,:]
    dist = arccos( sin(lat0)*sin(lats) + cos(lat0)*cos(lats)*cos( abs(lons-lon0) ) )    # r=1
    state.psi[:,:,0,:] = exp( -dist**2/(pi/50) )

    # velocity
    #alpha = 0       # zonal
    alpha = pi/2    # meridional

    state.velocity[:] = numpy.load('./preprocess_files/velocity.npy')
    

    """
    lons = lonlat_coord[0,:,:,:]
    lats = lonlat_coord[1,:,:,:]
    velocity[0,:,:,0,:] = 2*pi*(cos(lats)*cos(alpha) + sin(lats)*cos(lons)*sin(alpha))
    velocity[1,:,:,0,:] = - 2*pi*sin(lons)*sin(alpha)

    for ie in xrange(nelem):
        for gj in xrange(ngll):
            for gi in xrange(ngll):
                lon, lat = lonlat_coord[:,gi,gj,ie]

                if abs(lat - pi/2) < 1e-9:
                    velocity[:,gi,gj,0,ie] = 2*pi, 0
                elif abs(lat + pi/2) < 1e-9:
                    velocity[:,gi,gj,0,ie] = -2*pi, 0
                else:
                    velocity[0,gi,gj,0,ie] = 2*pi*(cos(lat)*cos(alpha) + sin(lat)*cos(lon)*sin(alpha))
                    velocity[1,gi,gj,0,ie] = - 2*pi*sin(lon)*sin(alpha)
                '''
                velocity[0,gi,gj,0,ie] = 2*pi*(cos(lat)*cos(alpha) + sin(lat)*cos(lon)*sin(alpha))
                velocity[1,gi,gj,0,ie] = - 2*pi*sin(lon)*sin(alpha)
                '''
    """


    #----------------------------------------------
    # spectral element
    #----------------------------------------------
    interact = InteractBetweenElems()
    inside = InsideElems(N, ngll, nelem, state, interact)

    # minimum dx, dt
    max_v = 2*pi
    dt = cfl*min_dx/max_v
    tloop = RungeKutta(dt, inside)
    tloop.allocate(state.psi.shape, state.psi.dtype)


    #----------------------------------------------
    # print the setup information
    #----------------------------------------------
    print '-'*47
    print 'N\t\t', N
    print 'ngll\t\t', ngll
    print 'nelem\t\t', nelem
    print 'cfl\t\t', cfl
    print 'min_dx\t\t', min_dx
    print 'dt\t\t', dt

    #tmax = int( numpy.ceil(1/dt) )
    tmax = 10
    tgap = 1

    print 'tmax\t\t', tmax
    print 'tgap\t\t', tgap
    print '-'*47
    print ''

    """
    ret_psi = inside.compute_rhs(0, state.psi)
    numpy.save('compute_rhs_psi0.npy', state.psi)
    numpy.save('compute_rhs_psi1.npy', ret_psi)

    numpy.save('./run/%.6d_rank%d_psi.npy' % (0, rank), state.psi)
    numpy.save('./run/%.6d_rank%d_velocity.npy' % (0, rank), state.velocity)
    """

    #----------------------------------------------
    # time loop
    #----------------------------------------------
    for tstep in xrange(1,tmax+1):
        tloop.update_rk4(state.psi)

        if tstep%tgap == 0:
            print 'tstep=\t%d/%d (%g %s)\r' % (tstep, tmax, tstep/tmax*100, '%'),
            sys.stdout.flush()

            #numpy.save('./run/%.6d_rank%d_psi.npy' % (tstep, rank), state.psi)

    print ''
