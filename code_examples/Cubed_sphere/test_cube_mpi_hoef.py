#------------------------------------------------------------------------------
# filename  : test_cube_mpi_hoef.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.12    start
#
# description:
#   Test the CubeMPI for the HOEF(High Order Elliptic Filter)  
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
from test_cube_mpi import pre_send, post_recv




def make_SPH_reference():
    '''
    Generate the SPH(Y25) and filtered fields: ne=30, ngq=4
    '''
    ne, ngq = 30, 4
    nproc, myrank = 1, 0

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubempi = CubeMPI(cubegrid, method='HOEF')


    #------------------------------------------------------------
    # Spherical harmonics Y25
    #------------------------------------------------------------
    lats = cubegrid.local_latlons[:,0]      # (local_ep_size,2)
    lons = cubegrid.local_latlons[:,1]
    Y25 = special.sph_harm(2, 5, lons, pi/2-lats).real
    filtered = np.zeros_like(Y25)


    #------------------------------------------------------------
    # High Order Elliptic Filter
    #------------------------------------------------------------
    send_buf = np.zeros(cubempi.send_buf_size, 'f8')
    recv_buf = np.zeros(cubempi.recv_buf_size, 'f8')
    pre_send(cubempi, Y25, recv_buf, send_buf)
    post_recv(cubempi, filtered, recv_buf)


    #------------------------------------------------------------
    # Save as NetCDF
    #------------------------------------------------------------
    ncf = nc.Dataset('Y25_and_HOEFed_ne%dngq%d.nc'%(ne,ngq), 'w', format='NETCDF4')
    ncf.description = 'HOEF(High Order Elliptic Filter) test: Spherical harmonics Y25 and filtered'
    ncf.date_of_production = '%s'%datetime.now()
    ncf.author = 'kh.kim@kiaps.org'
    ncf.ne = cubegrid.ne
    ncf.ngq = cubegrid.ngq
    ncf.createDimension('up_size', cubegrid.local_up_size)
    vy11 = ncf.createVariable('Y25', 'f8', ('up_size',))
    vfiltered = ncf.createVariable('filtered', 'f8', ('up_size',))

    vy11[:] = Y25[cubegrid.local_is_uvps]
    vfiltered[:] = filtered[cubegrid.local_is_uvps]
    ncf.close()




def check_hoef_mpi(ne, ngq, comm):
    '''
    CubeMPI for HOEF
    '''
    myrank = comm.Get_rank()
    nproc = comm.Get_size()

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubempi = CubeMPI(cubegrid, method='HOEF')


    # Generate a sequential field on the cubed-sphere
    lats = cubegrid.local_latlons[:,0]      # (local_ep_size,2)
    lons = cubegrid.local_latlons[:,1]
    Y25 = special.sph_harm(2, 5, lons, pi/2-lats).real
    filtered = np.zeros_like(Y25)

    # Apply the High-order Elliptic Filter
    send_buf = np.zeros(cubempi.send_buf_size, 'f8')
    recv_buf = np.zeros(cubempi.recv_buf_size, 'f8')

    # Send/Recv
    pre_send(cubempi, Y25, recv_buf, send_buf)

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

    # After receive
    post_recv(cubempi, filtered, recv_buf)


    #-----------------------------------------------------
    # Check if the filtered has same values with reference
    #-----------------------------------------------------
    local_up_sizes = comm.gather(cubegrid.local_up_size, root=0)

    if myrank == 0:
        sizes = np.cumsum(local_up_sizes)
        gids = np.zeros(sizes[-1], 'i4')
        f = np.zeros(sizes[-1], 'f8')

        gids[:sizes[0]] = cubegrid.local_gids[cubegrid.local_is_uvps]
        f[:sizes[0]] = filtered[cubegrid.local_is_uvps]

        for src in xrange(1,nproc):
            comm.Recv(gids[sizes[src-1]:sizes[src]], src, 10)

        for src in xrange(1,nproc):
            comm.Recv(f[sizes[src-1]:sizes[src]], src, 20)

        idxs = np.argsort(gids)
        ncf = nc.Dataset('Y25_and_HOEFed_ne30ngq4.nc', 'r', format='NETCDF4')
        aa_equal(f[idxs], ncf.variables['filtered'][:], 15)

    else:
        comm.Send(cubegrid.local_gids[cubegrid.local_is_uvps], 0, 10)
        comm.Send(filtered[cubegrid.local_is_uvps], 0, 20)




def run_mpi(ne, ngq, nproc):
    cmd = 'mpirun -np %d python test_cube_mpi_hoef.py %d %d'%(nproc, ne, ngq)
    proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    #print stdout
    assert stderr == '', stderr




def test_hoef_mpi():
    ne, ngq = 30, 4
    for nproc in [4,7,12]:
        func = with_setup(lambda:None, lambda:None)(lambda:run_mpi(ne,ngq,nproc)) 
        func.description = 'CubeMPI for HOEF: Spherical Harmonics(Y25) (ne=%d, ngq=%d, nproc=%d)'%(ne,ngq,nproc)
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
