#------------------------------------------------------------------------------
# filename  : compare_with_kim.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.14    start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_, with_setup

from cube_mpi import CubeGridMPI, CubeMPI


from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

assert nproc==16

ne, ngq = 30, 4
niter = 180
init_type = 'Y35'   # constant, Y35, cosine_bell

ncf = nc.Dataset('/home/kjsung/khkim_test/nproc%d_rank%d.nc'%(nproc,myrank), 'r', format='NETCDF3_CLASSIC')
field = ncf.variables['local_var'][:]

cubegrid = CubeGridMPI(ne, ngq, nproc, myrank, homme_style=True)
local_up_sizes = comm.gather(cubegrid.local_up_size, root=0)

if myrank == 0:
    print ''
    print 'ne=%d, ngq=%d, niter=%d, init_type=%s, nproc=%d'%(ne,ngq,niter,init_type,nproc)

    sizes = np.cumsum(local_up_sizes)
    gids = np.zeros(sizes[-1], 'i4')
    f = np.zeros(sizes[-1], 'f8')

    gids[:sizes[0]] = cubegrid.local_gids[cubegrid.local_is_uvps]
    f[:sizes[0]] = field[cubegrid.local_is_uvps]

    for src in xrange(1,nproc):
        comm.Recv(gids[sizes[src-1]:sizes[src]], src, 10)

    for src in xrange(1,nproc):
        comm.Recv(f[sizes[src-1]:sizes[src]], src, 20)

    ncf = nc.Dataset('hoef_%s_ne%dngq%d_600step.nc'%(init_type,ne,ngq), 'r', format='NETCDF4')
    idxs = np.argsort(gids)
    diff = f[idxs] - ncf.variables['field%d'%(niter)][:]
    print 'max', diff.max()
    print 'min', diff.min()
    #aa_equal(f[idxs]/1e4, ncf.variables['field%d'%(niter)][:]/1e4, 13)

else:
    comm.Send(cubegrid.local_gids[cubegrid.local_is_uvps], 0, 10)
    comm.Send(field[cubegrid.local_is_uvps], 0, 20)
