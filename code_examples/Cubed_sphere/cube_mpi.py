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
import numpy as np
import netCDF4 as nc
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal


from cube_partition import CubePartition


fname = __file__.split('/')[-1]
fdir = __file__.rstrip(fname)




class CubeMPI(object):
    def __init__(self, ne, ngq, nproc, myrank, spmat_fpath):
        self.ne = ne
        self.ngq = ngq
        self.nproc = nproc
        self.myrank = myrank
        self.spmat_fpath = spmat_fpath    # sparse matrix NetCDF file

        
        #-----------------------------------------------------
        # Read the grid and sparse matrix information
        #-----------------------------------------------------
        cs_fpath = fdir + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
        cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
        global_size = len( cs_ncf.dimensions['size'] )
        gq_indices = cs_ncf.variables['gq_indices'][:]

        spmat_ncf = nc.Dataset(spmat_fpath, 'r', format='NETCDF4')
        dsts = spmat_ncf.variables['dsts'][:]
        srcs = spmat_ncf.variables['srcs'][:]
        weights = spmat_ncf.variables['weights'][:]


        #-----------------------------------------------------
        # Set the rank and local indices
        #-----------------------------------------------------
        partition = CubePartition(ne, nproc)
        my_nelem = partition.nelems[myrank]

        lids = np.zeros(global_size, 'i4')
        ranks = np.zeros(global_size, 'i4')
        mygids = np.zeros(my_nelem*ngq*ngq, 'i4')

        lseqs = np.zeros(nproc, 'i4')
        for seq, (panel,ei,ej,gi,gj) in enumerate(gq_indices):
            proc = partition.elem_proc[panel-1,ei-1,ej-1]
            lid = lseqs[proc]
            lseqs[proc] += 1

            lids[seq] = lid
            ranks[seq] = proc

            if proc == myrank:
                mygids[lid] = seq

        a_equal(partition.nelems, lseqs//(ngq*ngq))


        #-----------------------------------------------------
        # Destination, source, weight from the sparse matrix
        # Make Generate the meta index grouped by rank
        #-----------------------------------------------------
        local_group = dict() # {dst:[(src,wgt),...]}
        send_group = dict()  # {rank:{dst:[(src,wgt),...]},...}
        recv_group = dict()  # {rank:{dst:[src,...]},...}

        for dst, src, wgt in zip(dsts, srcs, weights):
            if src in mygids and dst in mygids:
                if dst not in local_group.keys(): local_group[dst] = list()
                local_group[dst].append( (src,wgt) )

            elif src in mygids:
                dst_rank = ranks[dst]
                if dst_rank not in send_group.keys():
                    send_group[dst_rank] = dict()

                if dst not in send_group[dst_rank].keys():
                    send_group[dst_rank][dst] = list()

                send_group[dst_rank][dst].append( (src,wgt) )

            elif dst in mygids:
                src_rank = ranks[src]
                if src_rank not in recv_group.keys():
                    recv_group[src_rank] = dict()

                if dst not in recv_group[src_rank].keys():
                    recv_group[src_rank][dst] = list()

                recv_group[src_rank][dst].append(src)


        #-----------------------------------------------------
        # Make the send_schedule, send_dsts, send_srcs, send_wgts
        #-----------------------------------------------------
        send_schedule = np.zeros((len(send_group),3), 'i4')  #(rank,start,size)
        send_dsts, send_srcs, send_wgts = [], [], []

        local_buf_size = len(local_group)


        # send_schedule
        send_buf_seq = 0
        for seq, rank in enumerate( sorted(send_group.keys()) ):
            start = send_buf_seq
            size = len(send_group[rank])
            send_schedule[seq][:] = (rank, start, size)

            send_buf_seq += size

        send_buf_size = send_buf_seq
        send_buf = np.zeros(send_buf_size, 'i4')    # global dst index


        # send local indices in myrank
        # directly go to the recv_buf, not to the send_buf
        send_seq = 0
        for dst in sorted(local_group.keys()):
            for src, wgt in local_group[dst]:
                send_dsts.append(send_seq)      # buffer index
                send_srcs.append(lids[src])     # local index
                send_wgts.append(wgt)
            
            send_seq += 1


        # send indices for the other ranks
        for rank in sorted(send_group.keys()):
            for dst in sorted(send_group[rank].keys()):
                for src, wgt in send_group[rank][dst]:
                    send_dsts.append(send_seq)
                    send_srcs.append(lids[src])
                    send_wgts.append(wgt)

                send_buf[send_seq-local_buf_size] = dst
                send_seq += 1

        equal(send_buf_size, send_seq-local_buf_size)


        #-----------------------------------------------------
        # Make the recv_schedule, recv_dsts, recv_srcs
        #-----------------------------------------------------
        recv_schedule = np.zeros((len(recv_group),3), 'i4')  #(rank,start,size)
        recv_dsts, recv_srcs = [], []

        # recv_schedule
        recv_buf_seq = local_buf_size
        for seq, rank in enumerate( sorted(recv_group.keys()) ):
            start = recv_buf_seq
            size = len(recv_group[rank])
            recv_schedule[seq][:] = (rank, start, size)

            recv_buf_seq += size

        recv_buf_size = recv_buf_seq


        # recv indices
        recv_buf_list = sorted(local_group.keys())
        for rank in sorted(recv_group.keys()):
            recv_buf_list.extend( sorted(recv_group[rank].keys()) )

        recv_buf = np.array(recv_buf_list, 'i4')
        for dst in np.unique(recv_buf):
            for bsrc in np.where(recv_buf==dst)[0]:
                recv_dsts.append(lids[dst])     # local index
                recv_srcs.append(bsrc)          # buffer index

        equal(recv_buf_size, len(recv_buf))


        #-----------------------------------------------------
        # public variables for diagnostic
        #-----------------------------------------------------
        self.cs_ncf = cs_ncf
        self.spmat_ncf = spmat_ncf
        self.partition = partition

        self.ranks = ranks
        self.lids = lids

        self.local_group = local_group
        self.send_group = send_group
        self.recv_group = recv_group

        self.send_buf = send_buf    # global dst index
        self.recv_buf = recv_buf    # global dst index


        #-----------------------------------------------------
        # public variables for diagnostic
        #-----------------------------------------------------
        self.local_gids = mygids

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

    ne, ngq = 3, 4
    spmat_fpath = fdir + 'spmat_se_ne%dngq%d.nc'%(ne, ngq)

    nproc, myrank = 3, 2
    cubempi = CubeMPI(ne, ngq, nproc, myrank, spmat_fpath)
