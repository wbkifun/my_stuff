#------------------------------------------------------------------------------
# filename  : cube_mpi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.9      start
#             2015.9.16     sparse matrix based MPI organization
#             2015.10.2     split to CubeGridMPI and CubeMPI
#             2015.11.4     rename HOEF -> IMPVIS, apply read_netcdf_mpi
#             2015.11.12    add distribute_local_sparse_matrix()
#
#
# description: 
#   Generate the index tables for MPI parallel on the cubed-sphere
#------------------------------------------------------------------------------

from __future__ import division
from collections import OrderedDict
from itertools import groupby
import numpy as np
import netCDF4 as nc
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal

from cube_partition import CubePartition
from util.misc.log import logger
from util.grid.path import dir_cs_grid, dir_spmat




class CubeGridMPI(object):
    def __init__(self, ne, ngq, nproc, myrank, is_rotate=False, homme_style=False):
        self.ne = ne
        self.ngq = ngq
        self.nproc = nproc
        self.myrank = myrank

        if is_rotate:
            pass
        else:
            self.lat0 = 0
            self.lon0 = 0


        #-----------------------------------------------------
        # Read the grid indices
        #-----------------------------------------------------
        cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
        cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')

        ep_size = len( cs_ncf.dimensions['ep_size'] )
        up_size = len( cs_ncf.dimensions['up_size'] )
        gq_indices = cs_ncf.variables['gq_indices'][:]  # (ep_size,5)
        is_uvps = cs_ncf.variables['is_uvps'][:]        # (ep_size)
        uids = cs_ncf.variables['uids'][:]              # (ep_size)
        gids = cs_ncf.variables['gids'][:]              # (up_size)
        xyzs = cs_ncf.variables['xyzs'][:]              # (up_size,3)
        latlons = cs_ncf.variables['latlons'][:]        # (up_size,2)
        alpha_betas = cs_ncf.variables['alpha_betas'][:]# (ep_size,2)

        #mvps = cs_ncf.variables['mvps'][:]              # (ep_size,4)
        #nbrs = cs_ncf.variables['nbrs'][:]              # (up_size,8)


        #-----------------------------------------------------
        # Set the rank and local indices
        #-----------------------------------------------------
        logger.debug('Set the rank and local indices')

        partition = CubePartition(ne, nproc, homme_style)
        local_nelem = partition.nelems[myrank]

        local_ep_size = local_nelem*ngq*ngq

        local_gids = np.zeros(local_ep_size, 'i4')
        local_uids = np.zeros(local_ep_size, 'i4')
        local_is_uvps = np.zeros(local_ep_size, 'bool')
        local_gq_indices = np.zeros((local_ep_size,5), 'i4')
        local_alpha_betas = np.zeros((local_ep_size,2), 'f8')
        local_latlons = np.zeros((local_ep_size,2), 'f8')
        local_xyzs = np.zeros((local_ep_size,3), 'f8')


        # MPI rank at each grid point
        gq_eijs = gq_indices[:,:3] - 1
        idxs = gq_eijs[:,0]*ne*ne + gq_eijs[:,1]*ne + gq_eijs[:,2] 
        ranks = partition.elem_proc.ravel()[idxs]

        local_gids[:] = np.where(ranks == myrank)[0]
        local_is_uvps[:] = is_uvps[local_gids]
        local_gq_indices[:] = gq_indices[local_gids,:]
        local_alpha_betas[:] = alpha_betas[local_gids,:]

        local_uids[:] = uids[local_gids]
        local_xyzs[:] = xyzs[local_uids,:]
        local_latlons[:] = latlons[local_uids,:]


        # Local index at each grid point
        lids = np.zeros(ep_size, 'i4')
        for proc in xrange(nproc):
            local_ep_size_tmp = partition.nelems[proc]*ngq*ngq
            idxs = np.where(ranks == proc)[0]
            lids[idxs] = np.arange(local_ep_size_tmp, dtype='i4')


        #-----------------------------------------------------
        # Public variables
        #-----------------------------------------------------
        self.partition = partition
        self.ranks = ranks
        self.lids = lids

        self.ep_size = ep_size
        self.up_size = up_size
        self.local_ep_size = local_ep_size
        self.local_up_size = local_is_uvps.sum()
        self.local_nelem = local_nelem

        self.local_gids = local_gids
        self.local_uids = local_uids
        self.local_is_uvps = local_is_uvps
        self.local_gq_indices = local_gq_indices
        self.local_alpha_betas = local_alpha_betas
        self.local_latlons = local_latlons
        self.local_xyzs = local_xyzs




class CubeMPI(object):
    def __init__(self, cubegrid, method, comm=None):
        self.cubegrid = cubegrid
        self.method = method        # method represented by the sparse matrix

        self.ne = cubegrid.ne
        self.ngq = cubegrid.ngq
        self.nproc = cubegrid.nproc
        self.myrank = myrank = cubegrid.myrank
        self.ranks = cubegrid.ranks
        self.lids = cubegrid.lids

        if comm == None:
            self.read_sparse_matrix()
            self.arr_dict = self.extract_local_sparse_matrix(myrank)

        else:
            if myrank == 0:
                self.read_sparse_matrix()
                self.arr_dict = self.extract_local_sparse_matrix(0)

            self.distribute_local_sparse_matrix(comm)

        self.make_mpi_tables()



    def read_sparse_matrix(self):
        logger.debug('Read a NetCDF file as sparse matrix')

        ne = self.ne
        ngq = self.ngq
        method = self.method
        ranks = self.ranks

        if method.upper() == 'AVG':
            # Average the boundary of elements for the Spectral Element Method
            spmat_fpath = dir_spmat + 'spmat_avg_ne%dngq%d.nc'%(ne, ngq)

        elif method.upper() == 'COPY':
            # Copy from UP to EPs at the boundary of elements
            spmat_fpath = dir_spmat + 'spmat_copy_ne%dngq%d.nc'%(ne, ngq)

        elif method.upper() == 'IMPVIS':
            # Implicit Viscosity
            # High-Order Elliptic Filter
            spmat_fpath = dir_spmat + 'spmat_impvis_ne%dngq%d.nc'%(ne, ngq)

        else:
            raise ValueError, "The method must be one of 'AVG', 'COPY', 'IMPVIS'"

        spmat_ncf = nc.Dataset(spmat_fpath, 'r', format='NETCDF4')
        self.spmat_size = len( spmat_ncf.dimensions['spmat_size'] )
        self.dsts = spmat_ncf.variables['dsts'][:]
        self.srcs = spmat_ncf.variables['srcs'][:]
        self.wgts = spmat_ncf.variables['weights'][:]

        self.rank_dsts = ranks[self.dsts]   # rank number of destinations
        self.rank_srcs = ranks[self.srcs]   # rank number of sources



    def extract_local_sparse_matrix(self, target_rank):
        logger.debug('Extract local sparse matrix for rank%d'%target_rank)
        
        t_rank = target_rank
        dsts = self.dsts
        srcs = self.srcs
        wgts = self.wgts
        rank_dsts = self.rank_dsts
        rank_srcs = self.rank_srcs


        t_rank_dsts = (rank_dsts == t_rank)   # bool type array
        t_rank_srcs = (rank_srcs == t_rank)

        local_idxs = np.where( t_rank_dsts * t_rank_srcs )[0]
        send_idxs = np.where( np.invert(t_rank_dsts) * t_rank_srcs )[0]
        recv_idxs = np.where( t_rank_dsts * np.invert(t_rank_srcs) )[0]

        arr_dict = dict()
        arr_dict['spmat_size'] = self.spmat_size

        arr_dict['local_dsts'] = dsts[local_idxs]
        arr_dict['local_srcs'] = srcs[local_idxs]
        arr_dict['local_wgts'] = wgts[local_idxs]

        arr_dict['send_ranks'] = rank_dsts[send_idxs]
        arr_dict['send_dsts'] = dsts[send_idxs]
        arr_dict['send_srcs'] = srcs[send_idxs]
        arr_dict['send_wgts'] = wgts[send_idxs]

        arr_dict['recv_ranks'] = rank_srcs[recv_idxs]
        arr_dict['recv_dsts'] = dsts[recv_idxs]

        return arr_dict



    def distribute_local_sparse_matrix(self, comm):
        logger.debug('Distribute local sparse matrixes')

        if self.myrank == 0:
            req_list = list()
            
            for target_rank in xrange(1,nproc):
                arr_dict = self.extract_local_sparse_matrix(target_rank)
                req = comm.isend(arr_dict, dest=target_rank, tag=10)
                req_list.append(req)

            for req in req_list: req.wait()

        else:
            self.arr_dict = comm.recv(source=0, tag=10)



    def make_mpi_tables(self):
        '''
        Destination, source, weight from the sparse matrix
        Make Generate the meta index grouped by rank
        local_group: {dst:[(src,wgt),...]}
        send_group:  {rank:{dst:[(src,wgt),...]),...}
        recv_group:  {rank:[dst,...],...}
        All dictionaries are OrderedDicts.
        '''
        logger.debug('Make MPI tables')

        lids = self.lids
        arr_dict = self.arr_dict

        self.spmat_size = arr_dict['spmat_size']

        #---------------------------------------
        # local_group
        #---------------------------------------
        local_dsts = arr_dict['local_dsts']
        local_srcs = arr_dict['local_srcs']
        local_wgts = arr_dict['local_wgts']

        dsw_list = [(d,s,w) for d,s,w in zip(local_dsts,local_srcs,local_wgts)]
        local_group = OrderedDict([(dst, [(s,w) for (d,s,w) in val]) \
                for (dst, val) in groupby(dsw_list, lambda x:x[0])])

        local_src_size = len(dsw_list)
        local_buf_size = len(local_group)

        #---------------------------------------
        # send_group
        #---------------------------------------
        send_ranks = arr_dict['send_ranks']
        send_dsts = arr_dict['send_dsts']
        send_srcs = arr_dict['send_srcs']
        send_wgts = arr_dict['send_wgts']

        rdsw_list = [(r,d,s,w) for r,d,s,w in \
                zip(send_ranks,send_dsts,send_srcs,send_wgts)]

        sorted_rdsw_list = sorted(rdsw_list, key=lambda x:x[0])
        send_group_tmp = OrderedDict([(rank, [(d,s,w) for (r,d,s,w) in val]) \
                for (rank, val) in groupby(sorted_rdsw_list, lambda x:x[0])])

        send_group = OrderedDict()
        for rank, dsw_list in send_group_tmp.items():
            send_group[rank] = OrderedDict([(dst, [(s,w) for (d,s,w) in val]) \
                for (dst, val) in groupby(dsw_list, lambda x:x[0])])

        #---------------------------------------
        # recv_group
        #---------------------------------------
        recv_ranks = arr_dict['recv_ranks']
        recv_dsts = arr_dict['recv_dsts']

        rd_list = [(r,d) for r,d in zip(recv_ranks,recv_dsts)]

        sorted_rd_list = sorted(rd_list, key=lambda x:x[0])
        recv_group = OrderedDict([(rank, np.unique([d for (r,d) in val])) \
                for (rank, val) in groupby(sorted_rd_list, lambda x:x[0])])


        #-----------------------------------------------------
        # Make the send_schedule, send_dsts, send_srcs, send_wgts
        #-----------------------------------------------------
        logger.debug('Make the send_schedule, send_dsts, send_srcs, send_wgts')

        #---------------------------------------
        # size and allocation
        #---------------------------------------
        send_sche_size = len(send_group)
        send_buf_size = np.unique(send_dsts).size
        send_map_size = local_dsts.size + send_dsts.size

        send_schedule = np.zeros((send_sche_size,3), 'i4')  #(rank,start,size)
        send_dsts = np.zeros(send_map_size, 'i4')
        send_srcs = np.zeros(send_map_size, 'i4')
        send_wgts = np.zeros(send_map_size, 'f8')
        send_buf = np.zeros(send_buf_size, 'i4')    # global dst index

        #---------------------------------------
        # send_schedule
        #---------------------------------------
        send_buf_seq = 0
        for seq, rank in enumerate( send_group.keys() ):
            start = send_buf_seq
            size = len(send_group[rank])
            send_schedule[seq][:] = (rank, start, size)
            send_buf_seq += size

        if send_buf_size != send_buf_seq:
            logger.error("Error: send_buf_size(%d) != send_buf_seq(%d)"%(send_buf_size, send_buf_seq))
            raise SystemError

        #---------------------------------------
        # send local indices in myrank
        # directly go to the recv_buf, not to the send_buf
        #---------------------------------------
        seq = 0
        recv_buf_seq = 0
        for dst, sw_list in local_group.items():
            for src, wgt in sw_list:
                send_dsts[seq] = recv_buf_seq
                send_srcs[seq] = lids[src]
                send_wgts[seq] = wgt
                seq += 1

            recv_buf_seq += 1

        #---------------------------------------
        # send indices for the other ranks
        #---------------------------------------
        send_buf_seq = 0
        for rank, dst_dict in send_group.items():
            for dst, sw_list in dst_dict.items():
                for src, wgt in sw_list:
                    send_dsts[seq] = send_buf_seq
                    send_srcs[seq] = lids[src]
                    send_wgts[seq] = wgt
                    seq += 1

                send_buf[send_buf_seq] = dst     # for diagnostics
                send_buf_seq += 1

        if seq != send_map_size:
            logger.error("Error: seq(%d) != send_map_size(%d)"%(seq, send_map_size))
            raise SystemError

        if send_buf_seq != send_buf_size:
            logger.error("Error: send_buf_seq(%d) != send_buf_size(%d)"%(send_buf_seq, send_buf_size))
            raise SystemError

        #-----------------------------------------------------
        # Make the recv_schedule, recv_dsts, recv_srcs
        #-----------------------------------------------------
        logger.debug('Make the recv_schedule, recv_dsts, recv_srcs')

        #---------------------------------------
        # size and allocation
        #---------------------------------------
        recv_sche_size = len(recv_group)
        recv_buf_size = local_buf_size \
                + np.sum([d_unique.size for d_unique in recv_group.values()])
        recv_map_size = recv_buf_size

        recv_schedule = np.zeros((recv_sche_size,3), 'i4') #(rank,start,size)
        recv_dsts = np.zeros(recv_map_size, 'i4')
        recv_srcs = np.zeros(recv_map_size, 'i4')


        #---------------------------------------
        # recv_schedule
        #---------------------------------------
        recv_buf_seq = local_buf_size
        for seq, (rank,d_unique) in enumerate( recv_group.items() ):
            start = recv_buf_seq
            size = d_unique.size
            recv_schedule[seq][:] = (rank, start, size)
            recv_buf_seq += size

        #---------------------------------------
        # recv indices
        #---------------------------------------
        recv_buf_list = local_group.keys()      # destinations
        for rank, d_unique in recv_group.items():
            recv_buf_list.extend(d_unique)
        recv_buf = np.array(recv_buf_list, 'i4')

        unique_dsts = np.unique(recv_buf)
        seq = 0
        for dst in unique_dsts:
            for bsrc in np.where(recv_buf==dst)[0]:
                recv_dsts[seq] = lids[dst]      # local index
                recv_srcs[seq] = bsrc           # buffer index
                seq += 1


        #-----------------------------------------------------
        # Public variables for diagnostic
        #-----------------------------------------------------
        self.local_group = local_group
        self.send_group = send_group
        self.recv_group = recv_group

        self.send_buf = send_buf    # global dst index
        self.recv_buf = recv_buf    # global dst index


        #-----------------------------------------------------
        # Public variables
        #-----------------------------------------------------
        self.local_src_size = local_src_size
        self.send_buf_size = send_buf_size
        self.recv_buf_size = recv_buf_size

        self.send_schedule = send_schedule          # (rank,start,size)
        self.send_dsts = np.array(send_dsts, 'i4')  # to buffer
        self.send_srcs = np.array(send_srcs, 'i4')  # from local
        self.send_wgts = np.array(send_wgts, 'f8')

        self.recv_schedule = recv_schedule          # (rank,start,size)
        self.recv_dsts = np.array(recv_dsts, 'i4')  # to local
        self.recv_srcs = np.array(recv_srcs, 'i4')  # from buffer



    def save_netcdf(self, base_dir, target_method, nc_format):
        logger.debug('Save the mpi tables as NetCDF')

        ncf = nc.Dataset(base_dir + '/nproc%d_rank%d.nc'%(self.nproc,self.myrank), 'w', format=nc_format)   # 'NETCDF4', 'NETCDF3_CLASSIC'

        ncf.description = 'MPI index tables with the SFC partitioning on the cubed-sphere'
        ncf.target_method = target_method

        ncf.ne = self.ne
        ncf.ngq = self.ngq
        ncf.nproc = self.nproc
        ncf.myrank = self.myrank
        ncf.ep_size = self.cubegrid.ep_size
        ncf.up_size = self.cubegrid.up_size
        ncf.local_ep_size = self.cubegrid.local_ep_size
        ncf.local_up_size = self.cubegrid.local_up_size
        ncf.spmat_size = self.spmat_size

        ncf.createDimension('local_ep_size', self.cubegrid.local_ep_size)
        ncf.createDimension('send_sche_size', len(self.send_schedule))
        ncf.createDimension('recv_sche_size', len(self.recv_schedule))
        ncf.createDimension('send_size', len(self.send_dsts))
        ncf.createDimension('recv_size', len(self.recv_dsts))
        ncf.createDimension('3', 3)

        vlocal_gids = ncf.createVariable('local_gids', 'i4', ('local_ep_size',))
        vlocal_gids.long_name = 'Global index of local points'

        vbuf_sizes = ncf.createVariable('buf_sizes', 'i4', ('3',))
        vbuf_sizes.long_name = '[local_src_size, send_buf_size, recv_buf_size]'

        vsend_schedule = ncf.createVariable('send_schedule', 'i4', ('send_sche_size','3',))
        vsend_schedule.long_name = '[rank, start, size]'
        vrecv_schedule = ncf.createVariable('recv_schedule', 'i4', ('recv_sche_size','3',))
        vrecv_schedule.long_name = '[rank, start, size]'

        vsend_dsts = ncf.createVariable('send_dsts', 'i4', ('send_size',))
        vsend_dsts.long_name = 'Destination index for local and send buffer'
        vsend_srcs = ncf.createVariable('send_srcs', 'i4', ('send_size',))
        vsend_srcs.long_name = 'Source index for local and send buffer'
        vsend_wgts = ncf.createVariable('send_wgts', 'f8', ('send_size',))
        vsend_wgts.long_name = 'Weight value for local and send buffer'

        vrecv_dsts = ncf.createVariable('recv_dsts', 'i4', ('recv_size',))
        vrecv_dsts.long_name = 'Destination index for recv buffer'
        vrecv_srcs = ncf.createVariable('recv_srcs', 'i4', ('recv_size',))
        vrecv_srcs.long_name = 'Source index for recv buffer'


        vlocal_gids[:]    = self.cubegrid.local_gids[:]
        vbuf_sizes[:]     = (self.local_src_size, \
                             self.send_buf_size, \
                             self.recv_buf_size)
        vsend_schedule[:] = self.send_schedule[:]
        vrecv_schedule[:] = self.recv_schedule[:]
        vsend_dsts[:]     = self.send_dsts[:]
        vsend_srcs[:]     = self.send_srcs[:]
        vsend_wgts[:]     = self.send_wgts[:]
        vrecv_dsts[:]     = self.recv_dsts[:]
        vrecv_srcs[:]     = self.recv_srcs[:]

        ncf.close()




if __name__ == '__main__':
    import argparse
    import os
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    myrank = comm.Get_rank()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ne', type=int, help='number of elements')
    args = parser.parse_args()

    ngq = 4
    ne = args.ne
    #dpath = '/scratch/khkim/mpi_tables_ne%d_nproc%d'%(ne,nproc)    # GAON2
    dpath = './mpi_tables_ne%d_nproc%d'%(ne,nproc)

    if myrank == 0:
        print 'Generate the MPI tables for Implicit diffusion'
        print 'ne=%d, ngq=%d, target_nproc=%d'%(ne,ngq,nproc)

        if not os.path.exists(dpath):
            os.makedirs(dpath)

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank, homme_style=True)
    cubempi = CubeMPI(cubegrid, 'IMPVIS', comm)
    cubempi.save_netcdf(dpath, 'Implicit Viscosity', 'NETCDF3_CLASSIC')
