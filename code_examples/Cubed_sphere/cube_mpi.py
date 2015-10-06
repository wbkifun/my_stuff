#------------------------------------------------------------------------------
# filename  : cube_mpi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.9  start
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


fname = __file__.split('/')[-1]
fdir = __file__.rstrip(fname)




class CubeGridMPI(object):
    def __init__(self, ne, ngq, nproc, myrank):
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
        alpha_betas = cs_ncf.variables['alpha_betas'][:]# (up_size,2)
        latlons = cs_ncf.variables['latlons'][:]        # (up_size,2)
        xyzs = cs_ncf.variables['xyzs'][:]              # (up_size,3)

        #mvps = cs_ncf.variables['mvps'][:]              # (ep_size,4)
        #nbrs = cs_ncf.variables['nbrs'][:]              # (up_size,8)


        #-----------------------------------------------------
        # Set the rank and local indices
        #print 'Set the rank and local indices'
        #-----------------------------------------------------
        partition = CubePartition(ne, nproc)
        my_nelem = partition.nelems[myrank]

        local_ep_size = my_nelem*ngq*ngq

        local_gids = np.zeros(local_ep_size, 'i4')
        local_uids = np.zeros(local_ep_size, 'i4')
        local_is_uvps = np.zeros(local_ep_size, 'i2')
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

        local_uids = uids[local_gids]
        local_alpha_betas[:] = alpha_betas[local_uids,:]
        local_latlons[:] = latlons[local_uids,:]
        local_xyzs[:] = xyzs[local_uids,:]


        '''
        lseqs = np.zeros(nproc, 'i4')
        for seq, (panel,ei,ej,gi,gj) in enumerate(gq_indices):
            proc = partition.elem_proc[panel-1,ei-1,ej-1]
            lid = lseqs[proc]
            lseqs[proc] += 1

            if proc == myrank:
                local_gids[lid] = seq
                local_is_uvps[lid] = is_uvps[seq]
                local_gq_indices[lid,:] = gq_indices[seq,:]

                u_seq = uids[seq]
                local_alpha_betas[lid,:] = alpha_betas[u_seq,:]
                local_latlons[lid,:] = latlons[u_seq,:]
                local_xyzs[lid,:] = xyzs[u_seq,:]

        a_equal(partition.nelems, lseqs//(ngq*ngq))
        '''
        
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
        self.local_uids = local_gids
        self.local_is_uvps = local_is_uvps
        self.local_gq_indices = local_gq_indices
        self.local_alpha_betas = local_alpha_betas
        self.local_latlons = local_latlons
        self.local_xyzs = local_xyzs




class CubeMPI(object):
    def __init__(self, cubegrid, spmat_fpath):
        self.spmat_fpath = spmat_fpath    # sparse matrix NetCDF file

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
        #print 'Make Generate the meta index grouped by rank'
        #-----------------------------------------------------
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
        #print 'Make the send_schedule, send_dsts, send_srcs, send_wgts'
        #-----------------------------------------------------
        send_schedule = np.zeros((len(send_group),3), 'i4')  #(rank,start,size)
        local_buf_size = len(local_group)

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
        '''
        send_dsts = [seq for seq, sw_list in enumerate(local_group.values()) \
                         for src, wgt in sw_list]
        send_srcs = [lids[s] for sws in local_group.values() for s, w in sws]
        send_wgts = [w for sws in local_group.values() for s, w in sws]
        '''
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
        '''
        tmp_sizes = [0] + [len(d_dict) for d_dict in send_group.values()[:-1]]
        sizes = np.cumsum(tmp_sizes) + local_buf_size
        send_dsts += [seq1 + seq2 \
                for seq1, d_dict in zip(sizes, send_group.values()) \
                for seq2, sw_list in enumerate(d_dict.values()) \
                for src, wgt in sw_list]
        send_srcs += [lids[src] \
                for d_dict in send_group.values() \
                for sw_list in d_dict.values() \
                for src, wgt in sw_list]
        send_wgts += [wgt \
                for d_dict in send_group.values() \
                for sw_list in d_dict.values() \
                for src, wgt in sw_list]
        '''
        for rank, dst_dict in send_group.items():
            for dst, sw_list in dst_dict.items():
                for src, wgt in sw_list:
                    send_dsts.append(send_seq)
                    send_srcs.append(lids[src])
                    send_wgts.append(wgt)

                send_buf[send_seq-local_buf_size] = dst     # for diagnostics
                send_seq += 1

        equal(send_buf_size, send_dsts[-1]-local_buf_size+1)

        # for diagnostic
        '''
        send_buf[:] = [dst \
                for d_dict in send_group.values() \
                for dst, sw_list in d_dict.items()]
        '''


        #-----------------------------------------------------
        # Make the recv_schedule, recv_dsts, recv_srcs
        #print 'Make the recv_schedule, recv_dsts, recv_srcs'
        #-----------------------------------------------------
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

        '''
        recv_dsts = [lids[dst] \
                for dst in unique_dsts \
                for bsrc in np.where(recv_buf==dst)[0]]     # local index
        recv_srcs = [bsrc \
                for dst in unique_dsts \
                for bsrc in np.where(recv_buf==dst)[0]]     # buffer index

        '''
        recv_dsts, recv_srcs = [], []
        for dst in unique_dsts:
            for bsrc in np.where(recv_buf==dst)[0]:
                recv_dsts.append(lids[dst])     # local index
                recv_srcs.append(bsrc)          # buffer index

        #a_equal(recv_dsts2, recv_dsts)
        #a_equal(recv_srcs2, recv_srcs)


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
        self.local_gids = cubegrid.local_gids

        self.local_buf_size = local_buf_size
        self.send_buf_size = send_buf_size
        self.recv_buf_size = recv_buf_size

        self.send_schedule = send_schedule          # (rank,start,size)
        self.send_dsts = np.array(send_dsts, 'i4')  # to buffer
        self.send_srcs = np.array(send_srcs, 'i4')  # from local
        self.send_wgts = np.array(send_wgts, 'f8')

        self.recv_schedule = recv_schedule          # (rank,start,size)
        self.recv_dsts = np.array(recv_dsts, 'i4')  # to local
        self.recv_srcs = np.array(recv_srcs, 'i4')  # from buffer




if __name__ == '__main__':
    '''
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    '''

    ne, ngq = 30, 4
    spmat_fpath = fdir + 'spmat_se_ne%dngq%d.nc'%(ne, ngq)

    nproc, myrank = 1, 0
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubempi = CubeMPI(cubegrid, spmat_fpath)
