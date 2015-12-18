#------------------------------------------------------------------------------
# filename  : sem_sphere_2d.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.10.1     start
#
#
# description: 
#   Spectral Element Method on the sphere (2D)
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from math import fsum

from multi_platform.machine_platform import MachinePlatform
from multi_platform.array_variable import Array, ArrayAs
from cubed_sphere.cube_tensor import CubeTensor
from cubed_sphere.cube_mpi import CubeMPI




class SpectralElement2D(object):
    def __init__(self, platform, cubegrid, velocity):
        self.platform = platform
        self.cubegrid = cubegrid


        #--------------------------------------------------------
        # Transform matrix, Jacobian, Derivative matrix
        #--------------------------------------------------------
        cubetensor = CubeTensor(cubegrid)

        AI = ArrayAs(platform, cubetensor.AI.ravel(), 'AI') # (ep_size,2,2)
        J = ArrayAs(platform, cubetensor.J, 'J')            # (ep_size)
        dvvT = ArrayAs(platform, cubetensor.dvv.T.ravel(), 'dvvT') # (ngq,ngq)


        #--------------------------------------------------------
        # Prepare the update_element function
        #--------------------------------------------------------
        src = open(__file__.replace('.py','.'+platform.code_type)).read()
        pyf = open(__file__.replace('.py','.pyf')).read()

        lib = platform.source_compile(src, pyf)
        self.update_element_core = platform.get_function(lib, 'update_element')

        self.update_element_core.prepare('iiooooOO', cubegrid.local_ep_size, cubegrid.ngq, dvvT, J, AI, velocity)    # add (psi, ret_psi)
        

        #--------------------------------------------------------
        # Prepare the boundary average
        #--------------------------------------------------------
        import cubed_sphere
        ne = self.cubegrid.ne
        ngq = self.cubegrid.ngq

        sm_fpath = cubed_sphere.__file__.replace('__init__.py','spmat_se_ne%dngq%d.nc'%(ne,ngq))
        print 'cubempi'
        cubempi = CubeMPI(self.cubegrid, sm_fpath)
        print 'cubempi end'

        # Allocate send/recv buffers
        self.send_buf = np.zeros(cubempi.send_buf_size, 'f8')
        self.recv_buf = np.zeros(cubempi.recv_buf_size, 'f8')

        # Send
        self.send_dsts_unique, send_dsts_index = \
                np.unique(cubempi.send_dsts, return_index=True)
        self.send_dst_group = list(send_dsts_index) + [len(cubempi.send_dsts)]

        # Receive
        self.recv_dsts_unique, recv_index_dsts = \
                np.unique(cubempi.recv_dsts, return_index=True)
        self.recv_dst_group = list(recv_index_dsts) + [len(cubempi.recv_dsts)]



    def update_element(self, t, psi, ret_psi):
        self.update_element_core.prepared_call(psi, ret_psi)



    def average_boundary(self, psi):
        cubempi = self.cubempi
        send_buf = self.send_buf
        recv_buf = self.recv_buf


        #--------------------------------------------------------
        # Prepare the send buffer
        #--------------------------------------------------------
        local_buf_size = cubempi.local_buf_size
        wgts = cubempi.send_wgts
        send_srcs = cubempi.send_srcs
        send_dsts_unique = self.send_dsts_unique
        send_dst_group = self.send_dst_group

        for seq, u_dst in enumerate(send_dsts_unique):
            start, end = send_dst_group[seq], send_dst_group[seq+1]
            ws_list = [wgts[i]*psi.get()[send_srcs[i]] for i in xrange(start,end)]
            
            if seq < local_buf_size:
                recv_buf[u_dst] = fsum(ws_list)
            else:
                send_buf[u_dst-local_buf_size] = fsum(ws_list)


        #--------------------------------------------------------
        # Send/Recv communication
        #--------------------------------------------------------
        req_send_list = list()
        req_recv_list = list()

        for dest, start, size in cubempi.send_schedule:
            req = comm.Isend(send_buf[start:start+size], dest, 0)
            req_send_list.append(req)

        for dest, start, size in cubempi.recv_schedule:
            req = comm.Irecv(recv_buf[start:start+size], dest, 0)
            req_recv_list.append(req)

        MPI.Request.Waitall(req_send_list)
        MPI.Request.Waitall(req_recv_list)


        #--------------------------------------------------------
        # Linear summation from the recv buffer
        #--------------------------------------------------------
        recv_srcs = cubempi.recv_srcs
        recv_dsts_unique = self.recv_dsts_unique
        recv_dst_group = self.recv_dst_group

        for seq, u_dst in enumerate(recv_dsts_unique):
            start, end = recv_dst_group[seq], recv_dst_group[seq+1]
            ws_list = [recv_buf[recv_srcs[i]] for i in xrange(start,end)]
            psi.get()[u_dst] = fsum(ws_list)




if __name__ == '__main__':
    from numpy import pi, arccos, sin, cos, exp, fabs

    from multi_platform.machine_platform import MachinePlatform
    from multi_platform.array_variable import Array, ArrayAs
    from cubed_sphere.cube_mpi import CubeGridMPI
    from runge_kutta import RungeKutta


    #----------------------------------------------
    # Setup
    #----------------------------------------------
    ne = 30         # elements / axis
    ngq = 4         # GQ points / axis / element
    cfl = 0.1       # Courant-Friedrichs-Lewy condition 
    nproc = 1

    platform = MachinePlatform('CPU', 'f90')
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank=0)


    #----------------------------------------------
    # Variables
    #----------------------------------------------
    ep_size = cubegrid.local_ep_size
    psi = Array(platform, ep_size, 'f8', 'psi')
    velocity = Array(platform, ep_size*2, 'f8', 'velocity')


    #----------------------------------------------
    # Initialize the velocity vector and the scalar field
    #----------------------------------------------
    # Initialize with the 2D gaussian
    lon0, lat0 = pi/2, -pi/5
    latlons = cubegrid.local_latlons    # (local_ep_size,2)
    lats = latlons[:,0]
    lons = latlons[:,1]
    dist = arccos( sin(lat0)*sin(lats) + cos(lat0)*cos(lats)*cos( fabs(lons-lon0) ) )    # r=1
    psi.set( exp( -dist**2/(pi/50) ) )


    vel = np.zeros((ep_size,2), 'f8')

    #alpha = 0       # zonal
    alpha = pi/2    # meridional

    # vel[:,0] = 2*pi*(cos(lats)*cos(alpha) + sin(lats)*cos(lons)*sin(alpha))
    # vel[:,1] = - 2*pi*sin(lons)*sin(alpha)

    for seq, (lat, lon) in enumerate(latlons):
        if fabs(lat - pi/2) < 1e-9:
            vel[seq,:] = 2*pi, 0

        elif fabs(lat + pi/2) < 1e-9:
            vel[seq,:] = -2*pi, 0

        else:
            vel[seq,0] = 2*pi*(cos(lat)*cos(alpha) + sin(lat)*cos(lon)*sin(alpha))
            vel[seq,1] = - 2*pi*sin(lon)*sin(alpha)

    velocity.set( vel.ravel() )


    # minimum dx
    lat1, lon1 = latlons[0,:]
    lat2, lon2 = latlons[1,:]
    min_dx = arccos( sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos( abs(lon1-lon2) ) ) 

    # minimum dt
    max_v = 2*pi
    dt = cfl*min_dx/max_v


    #----------------------------------------------
    # spectral element
    #----------------------------------------------
    print 'a'
    se = SpectralElement2D(platform, cubegrid, velocity)
    print 'b'
    rk = RungeKutta(platform, ep_size, dt)
    print 'c'


    #----------------------------------------------
    # print the setup information
    #----------------------------------------------
    print '-'*47
    print 'ne\t\t', ne
    print 'ngq\t\t', ngq
    print 'cfl\t\t', cfl
    print 'min_dx\t\t', min_dx
    print 'dt\t\t', dt

    #tmax = int( numpy.ceil(1/dt) )
    tmax = 1000
    tgap = 10

    print 'tmax\t\t', tmax
    print 'tgap\t\t', tgap
    print '-'*47
    print ''

    """
    ret_psi = inside.compute_rhs_test(psi, 0)
    numpy.save('compute_rhs_psi0.npy', psi)
    numpy.save('compute_rhs_psi1.npy', ret_psi)

    """
    #numpy.save('./run/%.6d_rank%d_psi.npy' % (0, rank), psi)
    #numpy.save('./run/%.6d_rank%d_velocity.npy' % (0, rank), velocity)

    #----------------------------------------------
    # time loop
    #----------------------------------------------
    t = 0
    for tstep in xrange(1,tmax+1):
        rk.update_rk4(t, psi, se.update_element, se.average_boundary)
        t += dt

        if tstep%tgap == 0:
            print 'tstep=\t%d/%d (%g %s)\r' % (tstep, tmax, tstep/tmax*100, '%'),
            sys.stdout.flush()

            #numpy.save('./run/%.6d_rank%d_psi.npy' % (tstep, rank), psi)

    print ''
