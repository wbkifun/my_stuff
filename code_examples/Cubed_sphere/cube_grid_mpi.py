#------------------------------------------------------------------------------
# filename  : cube_grid_mpi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.11.10    replace broadcast with selection-push at master
#                           split to cube_grid_mpi.py and cube_mpi.py
#
#
# description: 
#   Generate the local grid indices of the cubed-sphere
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc

from cube_partition import CubePartition
from util.log import logger


tag = np.random.randint(1e6,1e7)




class ReadFileCubeGrid(object):
    def __init__(self, ne, ngq):
        logger.debug('Read a grid indices NetCDF file')

        self.ne = ne
        self.ngq = ngq

        fname = __file__.split('/')[-1]
        fdir = __file__.rstrip(fname)
        fpath = fdir + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
        self.ncf = ncf = nc.Dataset(fpath, 'r', format='NETCDF4')

        self.ep_size = len( ncf.dimensions['ep_size'] )
        self.up_size = len( ncf.dimensions['up_size'] )
        self.gq_indices = ncf.variables['gq_indices'][:]  # (ep_size,5)
        self.is_uvps = ncf.variables['is_uvps'][:]        # (ep_size)
        self.uids = ncf.variables['uids'][:]              # (ep_size)
        self.gids = ncf.variables['gids'][:]              # (up_size)
        self.xyzs = ncf.variables['xyzs'][:]              # (up_size,3)
        self.latlons = ncf.variables['latlons'][:]        # (up_size,2)
        self.alpha_betas = ncf.variables['alpha_betas'][:]# (ep_size,2)

        #self.mvps = ncf.variables['mvps'][:]              # (ep_size,4)
        #self.nbrs = ncf.variables['nbrs'][:]              # (up_size,8)




class CubeGridLocalIndices(object):
    def __init__(self, target_rank, csfile, partition, ranks):
        logger.debug('Make local indices of rank%d'%target_rank)

        local_nelem = partition.nelems[target_rank]
        local_ep_size = local_nelem*csfile.ngq*csfile.ngq

        local_gids = np.where(ranks == target_rank)[0]        # (local_ep_size,), int32
        local_is_uvps = csfile.is_uvps[local_gids]            # (local_ep_size,), bool
        local_gq_indices = csfile.gq_indices[local_gids,:]    # (local_ep_size,5), int32
        local_alpha_betas = csfile.alpha_betas[local_gids,:]  # (local_ep_size,2), float64

        local_uids = csfile.uids[local_gids]                  # (local_ep_size,), int32
        local_xyzs = csfile.xyzs[local_uids,:]                # (local_ep_size,3), float64
        local_latlons = csfile.latlons[local_uids,:]          # (local_ep_size,2), float64

        local_up_size = local_is_uvps.sum()


        # Packaging variables
        self.data_dict = data_dict = dict()

        data_dict['ne'] = csfile.ne
        data_dict['ngq'] = csfile.ngq
        data_dict['nproc'] = partition.nproc
        data_dict['myrank'] = target_rank

        data_dict['ep_size'] = csfile.ep_size
        data_dict['up_size'] = csfile.up_size 
        data_dict['local_ep_size'] = local_ep_size
        data_dict['local_up_size'] = local_up_size

        data_dict['local_gids'] = local_gids 
        data_dict['local_uids'] = local_uids
        data_dict['local_is_uvps'] = local_is_uvps
        data_dict['local_gq_indices'] = local_gq_indices
        data_dict['local_alpha_betas'] = local_alpha_betas
        data_dict['local_latlons'] = local_latlons
        data_dict['local_xyzs'] = local_xyzs




class CubeGridMPISlave(object):
    def __init__(self, comm, root=0):
        data_dict = comm.recv(source=root, tag=tag)

        for key, val in data_dict.items():
            setattr(self, key, val)




class CubeGridMPIMaster(object):
    def __init__(self, ne, ngq, nproc, homme_style=False):
        self.ne = ne
        self.ngq = ngq
        self.nproc = nproc

        # Partitioning
        partition = CubePartition(ne, nproc, homme_style)

        # Read a grid NetCDF file
        csfile = ReadFileCubeGrid(ne, ngq)

        # MPI rank at each grid point
        gq_eijs = csfile.gq_indices[:,:3] - 1
        idxs = gq_eijs[:,0]*ne*ne + gq_eijs[:,1]*ne + gq_eijs[:,2] 
        ranks = partition.elem_proc.ravel()[idxs]       # (ep_size,), int32

        # public variables
        self.partition = partition
        self.csfile = csfile
        self.ranks = ranks



    def send_to_slave(self, comm, target_rank, comm_rank):
        '''
        This routine is designed for working at a pseudo MPI environment.
        The target_rank is a real rank number.
        the comm_rank is only used for a communication.
        '''

        cgl = CubeGridLocalIndices(target_rank, \
                self.csfile, self.partition, self.ranks)

        logger.debug('Send local indices to rank%d'%target_rank)
        req = comm.isend(cgl.data_dict, dest=comm_rank, tag=tag)

        return req
