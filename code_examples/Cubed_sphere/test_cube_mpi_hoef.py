#------------------------------------------------------------------------------
# filename  : test_hoef.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.13    start
#             2015.10.14    Convert using MIP
#
# description:
#   Test the HOEF(High Order Elliptic Filter)  
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from math import fsum
from numpy import pi, sin, cos, tan, sqrt
from scipy import special
from datetime import datetime
import netCDF4 as nc
import subprocess as sp

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_, with_setup

from pkg.util.compare_float import feq

from cube_mpi import CubeGridMPI, CubeMPI
#from test_cube_mpi import pre_send, post_recv
from multi_platform.machine_platform import MachinePlatform
from multi_platform.array_variable import Array, ArrayAs




def make_HOEF_reference():
    '''
    Test the HOEF sparse matrix using the MIP(Machine Independent Platform)
    '''
    #-----------------------------------------------------------
    # Setup
    #-----------------------------------------------------------
    platform = MachinePlatform('CPU', 'f90')

    ne, ngq = 15, 4
    nproc, myrank = 1, 0
    niter = 600
    init_type = 'Y35'   # constant, Y35, cosine_bell

    print ''
    print 'ne=%d, ngq=%d, niter=%d, init_type=%s'%(ne,ngq,niter,init_type)

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)

    ep_size = cubegrid.local_ep_size
    lats = cubegrid.local_latlons[:,0]      # (local_ep_size,2)
    lons = cubegrid.local_latlons[:,1]
    theta, phi = lons, pi/2-lats

    if init_type == 'constant':
        init = np.ones(ep_size, 'f8')

    elif init_type == 'Y35':
        init = special.sph_harm(3, 5, theta, phi).real      # (m,l,theta,phi)
        #init = -(1/32)*sqrt(385/pi)*cos(3*theta)*sin(phi)**3*(9*cos(phi)**2-1)

    field1 = ArrayAs(platform, init)
    field2 = Array(platform, ep_size, 'f8')

    #-----------------------------------------------------------
    # Save as NetCDF
    #-----------------------------------------------------------
    ncf = nc.Dataset('hoef_%s_ne%dngq%d_%dstep_2.nc'%(init_type,ne,ngq,niter), 'w', format='NETCDF4')
    ncf.description = 'HOEF(High Order Elliptic Filter) test: %s field'%(init_type)
    ncf.date_of_production = '%s'%datetime.now()
    ncf.author = 'kh.kim@kiaps.org'
    ncf.ne = np.int32(cubegrid.ne)
    ncf.ngq = np.int32(cubegrid.ngq)
    ncf.createDimension('up_size', cubegrid.local_up_size)
    vfield0 = ncf.createVariable('field0', 'f8', ('up_size',))
    vfield0[:] = field1.get()[cubegrid.local_is_uvps]

    #-----------------------------------------------------------
    # Implicit filtering 
    #-----------------------------------------------------------
    spmat_fpath = './spmat_hoef_ne%dngq%d.nc'%(ne,ngq)
    spmat_ncf = nc.Dataset(spmat_fpath, 'r', format='NETCDF4')
    spmat_size = spmat_ncf.size
    dsts = ArrayAs(platform, spmat_ncf.variables['dsts'][:])
    srcs = ArrayAs(platform, spmat_ncf.variables['srcs'][:])
    wgts = ArrayAs(platform, spmat_ncf.variables['weights'][:])

    src = '''
SUBROUTINE imp_filter(ep_size, map_size, dsts, srcs, wgts, field1, field2)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: ep_size, map_size
  INTEGER, INTENT(IN) :: dsts(map_size), srcs(map_size)
  REAL(8), INTENT(IN) :: wgts(map_size), field1(ep_size)
  REAL(8), INTENT(INOUT) :: field2(ep_size)

  INTEGER :: i, dst, src
  REAL(8) :: wgt

  field2(:) = 0
  DO i=1,map_size
    dst = dsts(i) + 1
    src = srcs(i) + 1
    wgt = wgts(i)
    field2(dst) = field2(dst) + wgt*field1(src)
  END DO
END SUBROUTINE
    '''

    pyf = '''
python module $MODNAME
  interface
      subroutine imp_filter(ep_size, map_size, dsts, srcs, wgts, field1, field2)
      intent(c) :: imp_filter
      intent(c)   ! Adds to all following definitions
      integer, required, intent(in) :: ep_size, map_size
      integer, intent(in) :: dsts(map_size), srcs(map_size)
      real(8), intent(in) :: wgts(map_size), field1(ep_size)
      real(8), intent(inout) :: field2(ep_size)
    end subroutine
  end interface
end python module
    '''

    lib=platform.source_compile(src, pyf)
    imp_filter = platform.get_function(lib, 'imp_filter')
    imp_filter.prepare('iioooOO', ep_size, spmat_size, dsts, srcs, wgts)


    for seq in xrange(2,niter+1,2):
        imp_filter.prepared_call(field1, field2)

        # Append to the NetCDF
        if seq in [2,4,6,8,10]:
            print seq-1
            vfield = ncf.createVariable('field%d'%(seq-1), 'f8', ('up_size',))
            vfield[:] = field1.get()[cubegrid.local_is_uvps]

        imp_filter.prepared_call(field2, field1)

        if (seq//60)*60 == seq:
            print seq
            vfield = ncf.createVariable('field%d'%(seq), 'f8', ('up_size',))
            vfield[:] = field1.get()[cubegrid.local_is_uvps]

    ncf.close()




def check_hoef_mpi(ne, ngq, comm):
    '''
    CubeMPI for HOEF
    '''
    niter = 600
    init_type = 'Y35'   # constant, Y35, cosine_bell

    myrank = comm.Get_rank()
    nproc = comm.Get_size()

    platform = MachinePlatform('CPU', 'f90')
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubempi = CubeMPI(cubegrid, method='HOEF')


    #----------------------------------------------------------
    # Generate a initial field on the cubed-sphere
    #----------------------------------------------------------
    local_ep_size = cubegrid.local_ep_size
    lats = cubegrid.local_latlons[:,0]      # (local_ep_size,2)
    lons = cubegrid.local_latlons[:,1]
    theta, phi = lons, pi/2-lats

    if init_type == 'constant':
        init = np.ones(ep_size, 'f8')

    elif init_type == 'Y35':
        #init = special.sph_harm(3, 5, theta, phi).real      # (m,l,theta,phi)
        init = -(1/32)*sqrt(385/pi)*cos(3*theta)*sin(phi)**3*(9*cos(phi)**2-1)

    field = ArrayAs(platform, init)

    #----------------------------------------------------------
    # Apply the High-order Elliptic Filter with MPI
    #----------------------------------------------------------
    send_dsts = ArrayAs(platform, cubempi.send_dsts)
    send_srcs = ArrayAs(platform, cubempi.send_srcs)
    send_wgts = ArrayAs(platform, cubempi.send_wgts)
    recv_dsts = ArrayAs(platform, cubempi.recv_dsts)
    recv_srcs = ArrayAs(platform, cubempi.recv_srcs)
    send_buf = Array(platform, cubempi.send_buf_size, 'f8')
    recv_buf = Array(platform, cubempi.recv_buf_size, 'f8')

    src = open('cube_mpi.'+platform.code_type).read()
    pyf = open('cube_mpi.pyf').read()
    lib=platform.source_compile(src, pyf)
    pre_send = platform.get_function(lib, 'pre_send')
    post_recv = platform.get_function(lib, 'post_recv')
    pre_send.prepare('iiiiioooooo', \
            local_ep_size, send_dsts.size, cubempi.send_buf_size, \
            cubempi.recv_buf_size, cubempi.local_src_size, \
            send_dsts, send_srcs, send_wgts, field, send_buf, recv_buf)
    post_recv.prepare('iiioooo', \
            local_ep_size, recv_dsts.size, cubempi.recv_buf_size, \
            recv_dsts, recv_srcs, recv_buf, field)

    # Send/Recv
    for seq in xrange(niter):
        pre_send.prepared_call()

        req_send_list = list()
        req_recv_list = list()

        for dest, start, size in cubempi.send_schedule:
            req = comm.Isend(send_buf.get()[start:start+size], dest, 0)
            req_send_list.append(req)

        for dest, start, size in cubempi.recv_schedule:
            req = comm.Irecv(recv_buf.get()[start:start+size], dest, 0)
            req_recv_list.append(req)

        MPI.Request.Waitall(req_send_list)
        MPI.Request.Waitall(req_recv_list)

        # After receive
        post_recv.prepared_call()


    #-----------------------------------------------------
    # Check if the filtered has same values with reference
    #-----------------------------------------------------
    local_up_sizes = comm.gather(cubegrid.local_up_size, root=0)

    if myrank == 0:
        sizes = np.cumsum(local_up_sizes)
        gids = np.zeros(sizes[-1], 'i4')
        f = np.zeros(sizes[-1], 'f8')

        gids[:sizes[0]] = cubegrid.local_gids[cubegrid.local_is_uvps]
        f[:sizes[0]] = field.get()[cubegrid.local_is_uvps]

        for src in xrange(1,nproc):
            comm.Recv(gids[sizes[src-1]:sizes[src]], src, 10)

        for src in xrange(1,nproc):
            comm.Recv(f[sizes[src-1]:sizes[src]], src, 20)

        idxs = np.argsort(gids)
        ncf = nc.Dataset('hoef_%s_ne%dngq%d_600step.nc'%(init_type,ne,ngq), 'r', format='NETCDF4')
        aa_equal(f[idxs], ncf.variables['field%d'%(niter)][:], 14)

    else:
        comm.Send(cubegrid.local_gids[cubegrid.local_is_uvps], 0, 10)
        comm.Send(field.get()[cubegrid.local_is_uvps], 0, 20)




def run_mpi(ne, ngq, nproc):
    cmd = 'mpirun -np %d python test_cube_mpi_hoef.py %d %d'%(nproc, ne, ngq)
    proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    #print stdout
    assert stderr == '', stderr




def test_hoef_mpi():
    ne, ngq = 15, 4
    for nproc in [4,7,12,16]:
        func = with_setup(lambda:None, lambda:None)(lambda:run_mpi(ne,ngq,nproc)) 
        func.description = 'CubeMPI for HOEF: Spherical Harmonics(Y35) (ne=%d, ngq=%d, nproc=%d)'%(ne,ngq,nproc)
        yield func




if __name__ == '__main__':
    import argparse
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc = comm.Get_size()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('ngq', type=int, help='number of Gauss qudrature points')
    args = parser.parse_args()
    #if myrank == 0: print 'ne=%d, ngq=%d'%(args.ne, args.ngq) 

    check_hoef_mpi(args.ne, args.ngq, comm)
