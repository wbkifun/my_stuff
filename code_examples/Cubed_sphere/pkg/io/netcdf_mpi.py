#------------------------------------------------------------------------------
# filename  : netcdf_mpi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.11.4     start
#
#
# description: 
#   Integrated IO with NetCDF file on MPI environment
#------------------------------------------------------------------------------

from __future__ import division
import netCDF4 as nc
from mpi4py import MPI

from util.log import logger


comm = MPI.COMM_WORLD
nproc = comm.Get_size()
myrank = comm.Get_rank()




def read_netcdf_mpi(fpath, nc_dict, root, format='NETCDF4'):
    data_dict = dict()

    if myrank == root:
        for key in nc_dict.keys():
            if key not in ['dimension', 'variable']:
                logger.error("Error: wrong keyword of nc_dict, '%s')"%(key))

        ncf = nc.Dataset(fpath, 'r', format='NETCDF4')
        for dname in nc_dict.get('dimension',[]):
            data_dict[dname] = len( ncf.dimensions[dname] )

        for vname in nc_dict.get('variable',[]):
            data_dict[vname] = ncf.variables[vname][:]


    return comm.bcast(data_dict, root=root)
