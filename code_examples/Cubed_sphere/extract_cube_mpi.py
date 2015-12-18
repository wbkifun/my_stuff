#------------------------------------------------------------------------------
# filename  : extract_cube_mpi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.11.11    start
#
# description:
#   Extract variables in the cube_mpi.py to compare with KIM
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc

from cube_mpi import CubeGridMPI


ne, ngq = 30, 4
nproc = 16
myrank = 0

cubegrid = CubeGridMPI(ne, ngq, nproc, myrank, homme_style=True)

ncf = nc.Dataset('./extract_cube_mpi.nc', 'w', format='NETCDF3_CLASSIC')
ncf.description = 'Extracted variables in the cube_mpi.py to compare with KIM'
ncf.ne = ne
ncf.ngq = ngq
ncf.nproc = nproc
ncf.myrank = myrank

ncf.createDimension('ep_size', cubegrid.ep_size)
vlids = ncf.createVariable('lids', 'i4', ('ep_size',))
vlids[:] = cubegrid.lids[:]
ncf.close()


