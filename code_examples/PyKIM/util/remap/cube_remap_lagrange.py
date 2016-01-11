#------------------------------------------------------------------------------
# filename  : cube_remap_lagrange.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.8      start
#
#
# Description: 
#   Remap between cubed-sphere and latlon grid
#   using a Lagrange basis function
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import sys
from math import sqrt, pi, fsum

from util.convert_coord.cs_ll import latlon2abp




class LagrangeBasisFunction(object):
    def __init__(self, cs_obj, ll_obj):
        self.cs_obj = cs_obj
        self.ll_obj = ll_obj

        self.dst_size = ll_obj.nsize

        self.ngq = ngq = 4    # fixed
        self.mat_size = ngq*ngq



    def lagrange(self, order, x, xpts):
        ngq = self.ngq

        val = 1.0
        for i, xpt in enumerate(xpts):
            if i != order:
                val *= (x-xpt)/(xpts[order]-xpt)

        return val



    def get_weights(self, lat, lon, gids):
        '''
        Lagrange function: L(alpha)*L(beta)
        '''
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj
        ngq = self.ngq
        rlat, rlon = cs_obj.rlat, cs_obj.rlon

        panels = set([p for (p,ei,ej,gi,gj) in cs_obj.gq_indices[gids]])
        assert len(panels) == 1
        assert len(gids) == ngq*ngq, 'len(gids)=%d should be %d.'%(len(gids), ngq*ngq)

        weights = np.zeros(ngq*ngq, 'f8')

        panel = panels.pop()
        alpha, beta = latlon2abp(lat,lon,rlat,rlon)[panel]

        alpha_betas = cs_obj.alpha_betas[gids]
        alphas = alpha_betas[:ngq,0]
        betas = alpha_betas[::ngq,1]
        
        for j in xrange(ngq):
            for i in xrange(ngq):
                k = j*ngq + i
                weights[k] = self.lagrange(i,alpha,alphas) \
                            *self.lagrange(j,beta,betas)

        return weights



    def make_remap_matrix(self):
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj
        dst_size = self.dst_size
        mat_size = self.mat_size

        src_address = np.zeros((dst_size, mat_size), 'i4')
        remap_matrix = np.zeros((dst_size, mat_size), 'f8')

        for dst in xrange(ll_obj.nsize):
            lat, lon = ll_obj.latlons[dst]
            gids = cs_obj.get_surround_elem_gids(lat, lon)

            src_address[dst,:] = cs_obj.uids[gids]
            remap_matrix[dst,:] = self.get_weights(lat, lon, gids)

        return src_address, remap_matrix



    def make_remap_matrix_mpi(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        myrank = comm.Get_rank()

        if nproc == 1:
            return self.make_remap_matrix()

        cs_obj = self.cs_obj
        ll_obj = self.ll_obj
        mat_size = self.mat_size

        dst_size = ll_obj.nsize
        chunk_size = dst_size//nproc//10
        dsw_dict = dict()   # {dst:(srcs,wgts),...}


        if myrank == 0:
            start = 0
            while start < dst_size:
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send(start, dest=rank, tag=10)
                start += chunk_size

            for i in xrange(nproc-1):
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send('quit', dest=rank, tag=10)

                slave_dsw_dict = comm.recv(source=rank, tag=20)
                dsw_dict.update(slave_dsw_dict)


            src_address = np.zeros((dst_size, mat_size), 'i4')
            remap_matrix = np.zeros((dst_size, mat_size), 'f8')

            for dst in xrange(dst_size):
                srcs, wgts = dsw_dict[dst]
                src_address[dst,:] = srcs
                remap_matrix[dst,:] = wgts

            return src_address, remap_matrix


        else:
            while True:
                comm.send(myrank, dest=0, tag=0)
                msg = comm.recv(source=0, tag=10)

                if msg == 'quit':
                    print 'Slave rank %d quit.'%(myrank)
                    comm.send(dsw_dict, dest=0, tag=20)

                    return None, None

                start = msg
                end = start + chunk_size
                end = dst_size if end > dst_size else end
                print 'rank %d: %d ~ %d (%d %%)'% \
                        (myrank, start, end, end/dst_size*100)

                for dst in xrange(start,end):
                    lat, lon = ll_obj.latlons[dst]
                    gids = cs_obj.get_surround_elem_gids(lat, lon)

                    srcs = cs_obj.uids[gids]
                    wgts = self.get_weights(lat, lon, gids)
                    dsw_dict[dst] = (srcs,wgts)



    def set_netcdf_remap_matrix(self, ncf, src_address, remap_matrix):
        ncf.createDimension('dst_size', self.dst_size)
        ncf.createDimension('mat_size', self.mat_size)

        vsrc_address = ncf.createVariable('src_address', 'i4', ('dst_size','mat_size'))
        vremap_matrix = ncf.createVariable('remap_matrix', 'f8', ('dst_size','mat_size'))

        vsrc_address[:] = src_address[:]
        vremap_matrix[:] = remap_matrix[:]
