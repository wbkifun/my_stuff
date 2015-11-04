#------------------------------------------------------------------------------
# filename  : cube_mpi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.9      start
#             2015.9.16     sparse matrix based MPI organization
#             2015.10.2     split to CubeGridMPI and CubeMPI
#             2015.11.4     rename HOEF -> IMPVIS
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
from util.log import logger


fname = __file__.split('/')[-1]
fdir = __file__.rstrip(fname)




class CubeGridMPI(object):
    def __init__(self, ne, ngq, nproc, myrank, homme_style=False):
        self.ne = ne
        self.ngq = ngq
        self.nproc = nproc
        self.myrank = myrank


        #-----------------------------------------------------
        # Read the grid indices
        #-----------------------------------------------------
        cs_fpath = fdir + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
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
        my_nelem = partition.nelems[myrank]

        local_ep_size = my_nelem*ngq*ngq

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


        #-----------------------------------------------------
        # Public variables
        #-----------------------------------------------------
        self.cs_ncf = cs_ncf
        self.partition = partition

        self.ranks = ranks

        self.ep_size = ep_size
        self.up_size = up_size
        self.local_ep_size = local_ep_size
        self.local_up_size = local_is_uvps.sum()

        self.local_gids = local_gids
        self.local_uids = local_uids
        self.local_is_uvps = local_is_uvps
        self.local_gq_indices = local_gq_indices
        self.local_alpha_betas = local_alpha_betas
        self.local_latlons = local_latlons
        self.local_xyzs = local_xyzs




class CubeMPI(object):
    def __init__(self, cubegrid, method):
        self.cubegrid = cubegrid
        self.method = method        # method represented by the sparse matrix

        self.ne = ne = cubegrid.ne
        self.ngq = ngq = cubegrid.ngq
        self.nproc = nproc = cubegrid.nproc
        self.myrank = myrank = cubegrid.myrank
        self.ranks = ranks = cubegrid.ranks


        # Local index at each grid point
        lids = np.zeros(cubegrid.ep_size, 'i4')
        for proc in xrange(nproc):
            local_ep_size = cubegrid.partition.nelems[proc]*ngq*ngq
            idxs = np.where(ranks == proc)[0]
            lids[idxs] = np.arange(local_ep_size, dtype='i4')
        

        #-----------------------------------------------------
        # Read the sparse matrix
        #-----------------------------------------------------
        if method.upper() == 'AVG':
            # Average the boundary of elements for the Spectral Element Method
            spmat_fpath = fdir + 'spmat_avg_ne%dngq%d.nc'%(ne, ngq)

        elif method.upper() == 'COPY':
            # Copy from UP to EPs at the boundary of elements
            spmat_fpath = fdir + 'spmat_copy_ne%dngq%d.nc'%(ne, ngq)

        elif method.upper() == 'IMPVIS':
            # Implicit Viscosity
            # High-Order Elliptic Filter
            spmat_fpath = fdir + 'spmat_impvis_ne%dngq%d.nc'%(ne, ngq)

        else:
            raise ValueError, "The method must be one of 'AVG', 'COPY', 'IMPVIS'"

        spmat_ncf = nc.Dataset(spmat_fpath, 'r', format='NETCDF4')
        dsts = spmat_ncf.variables['dsts'][:]
        srcs = spmat_ncf.variables['srcs'][:]
        wgts = spmat_ncf.variables['weights'][:]


        #-----------------------------------------------------
        # Destination, source, weight from the sparse matrix
        # Make Generate the meta index grouped by rank
        # local_group: {dst:[(src,wgt),...]}
        # send_group:  {rank:{dst:[(src,wgt),...]),...}
        # recv_group:  {rank:{dst:[src,...]),...}
        # All dictionaries are OrderedDicts.
        #-----------------------------------------------------
        logger.debug('Make Generate the meta index grouped by rank')

        rank_dsts = ranks[dsts]                 # rank number of destinations
        rank_srcs = ranks[srcs]                 # rank number of sources
        myrank_dsts = (rank_dsts == myrank)     # bool type array
        myrank_srcs = (rank_srcs == myrank)

        local_idxs = np.where( myrank_dsts * myrank_srcs )[0]
        send_idxs = np.where( np.invert(myrank_dsts) * myrank_srcs )[0]
        recv_idxs = np.where( myrank_dsts * np.invert(myrank_srcs) )[0]

        #---------------------------------------
        # local_group
        #---------------------------------------
        dsw_list = [(dsts[i],srcs[i],wgts[i]) for i in local_idxs]
        local_group = OrderedDict([(dst, [(s,w) for (d,s,w) in val]) \
                for (dst, val) in groupby(dsw_list, lambda x:x[0])])
        local_src_size = len(dsw_list)
        local_buf_size = len(local_group)

        #---------------------------------------
        # send_group
        #---------------------------------------
        rdsw_list = [(rank_dsts[i],dsts[i],srcs[i],wgts[i]) for i in send_idxs]
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
        rds_list = [(rank_srcs[i],dsts[i],srcs[i]) for i in recv_idxs]
        sorted_rds_list = sorted(rds_list, key=lambda x:x[0])
        recv_group_tmp = OrderedDict([(rank, [(d,s) for (r,d,s) in val]) \
                for (rank, val) in groupby(sorted_rds_list, lambda x:x[0])])

        recv_group = OrderedDict()
        for rank, ds_list in recv_group_tmp.items():
            recv_group[rank] = OrderedDict([(dst, [s for (d,s) in val]) \
                for (dst, val) in groupby(ds_list, lambda x:x[0])])


        #-----------------------------------------------------
        # Make the send_schedule, send_dsts, send_srcs, send_wgts
        #-----------------------------------------------------
        logger.debug('Make the send_schedule, send_dsts, send_srcs, send_wgts')

        send_schedule = np.zeros((len(send_group),3), 'i4')  #(rank,start,size)

        #---------------------------------------
        # send_schedule
        #---------------------------------------
        send_buf_seq = 0
        for seq, rank in enumerate( sorted(send_group.keys()) ):
            start = send_buf_seq
            size = len(send_group[rank])
            send_schedule[seq][:] = (rank, start, size)
            send_buf_seq += size

        send_buf_size = send_buf_seq
        send_buf = np.zeros(send_buf_size, 'i4')    # global dst index

        #---------------------------------------
        # send local indices in myrank
        # directly go to the recv_buf, not to the send_buf
        #---------------------------------------
        send_dsts, send_srcs, send_wgts = list(), list(), list()
        send_seq = 0
        for dst, sw_list in local_group.items():
            for src, wgt in sw_list:
                send_dsts.append(send_seq)      # buffer index
                send_srcs.append(lids[src])     # local index
                send_wgts.append(wgt)
            send_seq += 1

        #---------------------------------------
        # send indices for the other ranks
        #---------------------------------------
        send_seq = 0
        for rank, dst_dict in send_group.items():
            for dst, sw_list in dst_dict.items():
                for src, wgt in sw_list:
                    send_dsts.append(send_seq)
                    send_srcs.append(lids[src])
                    send_wgts.append(wgt)

                send_buf[send_seq] = dst     # for diagnostics
                send_seq += 1


        #-----------------------------------------------------
        # Make the recv_schedule, recv_dsts, recv_srcs
        #-----------------------------------------------------
        logger.debug('Make the recv_schedule, recv_dsts, recv_srcs')

        recv_schedule = np.zeros((len(recv_group),3), 'i4')  #(rank,start,size)


        #---------------------------------------
        # recv_schedule
        #---------------------------------------
        recv_buf_seq = local_buf_size
        for seq, rank in enumerate( sorted(recv_group.keys()) ):
            start = recv_buf_seq
            size = len(recv_group[rank])
            recv_schedule[seq][:] = (rank, start, size)
            recv_buf_seq += size

        recv_buf_size = recv_buf_seq


        # recv indices
        recv_buf_list = local_group.keys()      # destinations
        for rank in recv_group.keys():
            recv_buf_list.extend( recv_group[rank].keys() )

        recv_buf = np.array(recv_buf_list, 'i4')
        equal(recv_buf_size, len(recv_buf))
        unique_dsts = np.unique(recv_buf)

        recv_dsts, recv_srcs = [], []
        for dst in unique_dsts:
            for bsrc in np.where(recv_buf==dst)[0]:
                recv_dsts.append(lids[dst])     # local index
                recv_srcs.append(bsrc)          # buffer index


        #-----------------------------------------------------
        # Public variables for diagnostic
        #-----------------------------------------------------
        self.spmat_ncf = spmat_ncf
        self.lids = lids

        self.local_group = local_group
        self.send_group = send_group
        self.recv_group = recv_group

        self.send_buf = send_buf    # global dst index
        self.recv_buf = recv_buf    # global dst index


        #-----------------------------------------------------
        # Public variables
        #-----------------------------------------------------
        self.spmat_size = len( spmat_ncf.dimensions['spmat_size'] )
        self.local_gids = cubegrid.local_gids

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



    def save_netcdf(self, base_dir, target_method):
        #ncf = nc.Dataset(base_dir + '/nproc%d_rank%d.nc'%(self.nproc,self.myrank), 'w', format='NETCDF4')
        ncf = nc.Dataset(base_dir + '/nproc%d_rank%d.nc'%(self.nproc,self.myrank), 'w', format='NETCDF3_CLASSIC')
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


        vlocal_gids[:]    = self.local_gids[:]
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
    parser.add_argument('target_nproc', type=int, help='number of target MPI processes')
    args = parser.parse_args()

    ngq = 4
    ne = args.ne
    target_nproc = args.target_nproc

    if myrank == 0:
        print 'Generate the MPI tables for Implicit diffusion'
        print 'ne=%d, ngq=%d, target_nproc=%d'%(ne,ngq,target_nproc)

        dpath = './mpi_tables_ne%d_nproc%d'%(ne,target_nproc)
        if not os.path.exists(dpath):
            os.makedirs(dpath)

        for target_proc in xrange(target_nproc):
            rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
            comm.send(target_proc, dest=rank, tag=10)

        for proc in xrange(nproc-1):
            rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
            comm.send('quit', dest=rank, tag=10)

    else:
        while True:
            comm.send(myrank, dest=0, tag=0)
            target_proc = comm.recv(source=0, tag=10)

            if target_proc == 'quit':
                print 'rank %d quit'%(myrank)
                break
            else:
                print 'target_rank: %d'%(target_proc)
                cubegrid = CubeGridMPI(ne, ngq, target_nproc, target_proc, homme_style=True)
                cubempi = CubeMPI(cubegrid, 'IMPVIS')
                cubempi.save_netcdf('./mpi_tables_ne%d_nproc%d'%(ne,target_nproc), 'Implicit Viscosity')
