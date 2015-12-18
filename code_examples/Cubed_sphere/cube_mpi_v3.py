#------------------------------------------------------------------------------
# filename  : cube_mpi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.9      start
#             2015.9.16     sparse matrix based MPI organization
#             2015.10.2     split to CubeGridMPI and CubeMPI
#             2015.11.4     rename HOEF -> IMPVIS, apply read_netcdf_mpi
#             2015.11.10    replace broadcast with selection-push at master
#                           split to cube_grid_mpi.py and cube_mpi.py
#
#
# description: 
#   Generate the index tables for MPI parallel on the cubed-sphere
#------------------------------------------------------------------------------

from __future__ import division
from collections import OrderedDict
from itertools import groupby
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
import numpy as np
import netCDF4 as nc

from util.log import logger


tag = np.random.randint(1e6,1e7)



class ReadFileSparseMatrix(object):
    def __init__(self, ne, ngq, method):
        logger.debug('Read a sparse matrix NetCDF file')

        self.ne = ne
        self.ngq = ngq
        self.method = method
        
        fname = __file__.split('/')[-1]
        fdir = __file__.rstrip(fname)

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

        self.ncf = ncf = nc.Dataset(spmat_fpath, 'r', format='NETCDF4')
        self.dsts = ncf.variables['dsts'][:]
        self.srcs = ncf.variables['srcs'][:]
        self.wgts = ncf.variables['weights'][:]




class LocalMPITables(object):
    def __init__(self, myrank, spfile, ranks, lids):
        # Classify Destination, source, weight from the sparse matrix
        logger.debug('Classify the meta indices grouped for rank%d'%myrank)

        dsts = spfile.dsts
        srcs = spfile.srcs
        wgts = spfile.wgts

        rank_dsts = ranks[dsts]                 # rank number of destinations
        rank_srcs = ranks[srcs]                 # rank number of sources
        myrank_dsts = (rank_dsts == myrank)     # bool type array
        myrank_srcs = (rank_srcs == myrank)

        local_idxs = np.where( myrank_dsts * myrank_srcs )[0]
        send_idxs = np.where( np.invert(myrank_dsts) * myrank_srcs )[0]
        recv_idxs = np.where( myrank_dsts * np.invert(myrank_srcs) )[0]

        dsw_list = [(dsts[i],srcs[i],wgts[i]) for i in local_idxs]
        rdsw_list = [(rank_dsts[i],dsts[i],srcs[i],wgts[i]) for i in send_idxs]
        rds_list = [(rank_srcs[i],dsts[i],srcs[i]) for i in recv_idxs]


        # packaging variables
        self.data_dict = data_dict = dict()

        data_dict['myrank'] = myrank
        data_dict['lids'] = lids
        data_dict['dsw_list'] = dsw_list
        data_dict['rdsw_list'] = rdsw_list
        data_dict['rds_list'] = rds_list




class CubeMPISlave(object):
    def __init__(self, comm, root=0):
        #data_dict = comm.recv(source=root, tag=tag)
        req = comm.irecv(source=root, tag=tag)
        data_dict = req.wait()

        myrank = data_dict['myrank']
        lids = data_dict['lids']
        dsw_list = data_dict['dsw_list']
        rdsw_list = data_dict['rdsw_list']
        rdw_list = data_dict['rds_list']


        #-----------------------------------------------------
        # Destination, source, weight from the sparse matrix
        # Make Generate the meta index grouped by rank
        # local_group: {dst:[(src,wgt),...]}
        # send_group:  {rank:{dst:[(src,wgt),...]),...}
        # recv_group:  {rank:{dst:[src,...]),...}
        #-----------------------------------------------------

        #---------------------------------------
        # local_group
        #---------------------------------------
        logger.debug('Make a local_group')

        local_group = OrderedDict([(dst, [(s,w) for (d,s,w) in val]) \
                for (dst, val) in groupby(dsw_list, lambda x:x[0])])
        local_src_size = len(dsw_list)
        local_buf_size = len(local_group)

        #---------------------------------------
        # send_group
        #---------------------------------------
        logger.debug('Make a send_group')

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
        logger.debug('Make the recv_group')

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
        self.local_group = local_group
        self.send_group = send_group
        self.recv_group = recv_group

        self.send_buf = send_buf    # global dst index
        self.recv_buf = recv_buf    # global dst index


        #-----------------------------------------------------
        # Public variables
        #-----------------------------------------------------
        self.myrank         = myrank

        self.local_src_size = local_src_size
        self.send_buf_size  = send_buf_size
        self.recv_buf_size  = recv_buf_size
                                                         
        self.send_schedule  = send_schedule              # (rank,start,size(
        self.send_dsts      = np.array(send_dsts, 'i4')  # to buffer   
        self.send_srcs      = np.array(send_srcs, 'i4')  # from local  
        self.send_wgts      = np.array(send_wgts, 'f8')                
                                                                           
        self.recv_schedule  = recv_schedule              # (rank,start,size)
        self.recv_dsts      = np.array(recv_dsts, 'i4')  # to local    
        self.recv_srcs      = np.array(recv_srcs, 'i4')  # from buffer 



    def save_netcdf(self, cubegrid, base_dir, target_method, nc_format):
        logger.debug('Save to netcdf of MPI tables of rank%d'%self.myrank)

        ncf = nc.Dataset(base_dir + '/nproc%d_rank%d.nc'%(self.nproc,self.myrank), 'w', format=nc_format)
        ncf.description = 'MPI index tables with the SFC partitioning on the cubed-sphere'
        ncf.target_method = target_method

        ncf.ne = cubegrid.ne
        ncf.ngq = cubegrid.ngq
        ncf.nproc = cubegrid.nproc
        ncf.myrank = self.myrank
        ncf.ep_size = cubegrid.ep_size
        ncf.up_size = cubegrid.up_size
        ncf.local_ep_size = cubegrid.local_ep_size
        ncf.local_up_size = cubegrid.local_up_size

        ncf.createDimension('local_ep_size', cubegrid.local_ep_size)
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

        logger.debug('Save to netcdf of MPI tables of rank%d...OK'%self.myrank)




class CubeMPIMaster(object):
    def __init__(self, cubegrid, method):
        self.ne = ne = cubegrid.ne
        self.ngq = ngq = cubegrid.ngq
        self.nproc = nproc = cubegrid.nproc
        self.method = method        # method represented by the sparse matrix

        ranks = cubegrid.ranks

        # Read the sparse matrix
        spfile = ReadFileSparseMatrix(ne, ngq, method)

        # Local index at each grid point
        lids = np.zeros(ranks.size, 'i4')
        for proc in xrange(nproc):
            local_ep_size = cubegrid.partition.nelems[proc]*ngq*ngq
            idxs = np.where(ranks == proc)[0]
            lids[idxs] = np.arange(local_ep_size, dtype='i4')

        # public variables
        self.spfile = spfile
        self.ranks = ranks
        self.lids = lids



    def send_to_slave(self, comm, target_rank, comm_rank):
        '''
        This routine is designed to work pseudo MPI environment.
        The target_rank is a real rank number.
        the comm_rank is only used for a communication.
        '''

        mtl = LocalMPITables(target_rank, self.spfile, self.ranks, self.lids)

        logger.debug('Send local tables to rank%d'%target_rank)
        req = comm.isend(mtl.data_dict, dest=comm_rank, tag=tag)
        logger.debug('after Send local tables to rank%d'%target_rank)

        return req
