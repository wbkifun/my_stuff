#------------------------------------------------------------------------------
# filename  : cube_remap_bilinear.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.16    start
#             2016.1.4      add make_remap_matrix_cs2ll()
#                           bugfix negative weight at make_remap_matrix_ll2cs()
#             2016.1.14     change threshold from 1e-13 to 1e-10 empirically
#
#
# Description: 
#   Remap between cubed-sphere and latlon grid
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import sys
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal

from util.convert_coord.cs_ll import latlon2abp
from util.misc.compare_float import flge




class Bilinear(object):
    def __init__(self, cs_obj, ll_obj, direction):
        self.cs_obj = cs_obj
        self.ll_obj = ll_obj
        self.direction = direction

        assert direction in ['ll2cs', 'cs2ll', 'cs2cs', 'cs2aws'], "{} direction is not supported.".format(direction)

        if direction == 'll2cs':
            self.dst_obj = cs_obj
        else:
            self.dst_obj = ll_obj

        self.dst_size = self.dst_obj.nsize
        self.mat_size = 4



    def get_bilinear_weights(self, dst, xy, xy1, xy2):
        x, y = xy
        x1, y1 = xy1
        x2, y2 = xy2

        dx, dy = x2-x1, y2-y1
        dxy = dx*dy

        w1 = (x2-x)*(y2-y)/dxy
        w2 = (x-x1)*(y2-y)/dxy
        w3 = (x2-x)*(y-y1)/dxy
        w4 = (x-x1)*(y-y1)/dxy

        weights = np.zeros(4, 'f8')
        for i, w in enumerate([w1,w2,w3,w4]): 
            if np.fabs(w) < 1e-10:
                w = 0

            if w < 0:
                print("dst: {}".format(dst))
                print("x,y: {}, {}".format(x,y))
                print("x1,x2: {}, {}".format(x1,x2))
                print("y1,y2: {}, {}".format(y1,y2))
                print("dx,dy: {}, {}".format(dx,dy))
                print("weights: {}, {}, {}, {}".format(w1,w2,w3,w4))
                sys.exit()

            weights[i] = w

        return weights



    def get_weights_ll2cs(self, dst, lat, lon, idxs):
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj

        idx1, idx2, idx3, idx4 = idxs

        lat1, lon1 = ll_obj.latlons[idx1]
        lat2, lon2 = ll_obj.latlons[idx2]
        lat3, lon3 = ll_obj.latlons[idx3]
        lat4, lon4 = ll_obj.latlons[idx4]

        assert np.fabs(lon1-lon3)<1e-15
        assert np.fabs(lon2-lon4)<1e-15
        assert np.fabs(lat1-lat2)<1e-15
        assert np.fabs(lat3-lat4)<1e-15

        if lon2 < lon1: lon2 = lon1 + ll_obj.dlon
        if lon4 < lon3: lon4 = lon3 + ll_obj.dlon
        if np.fabs(lon-lon1) > np.pi: lon += 2*np.pi
        assert flge(lon1,lon,lon2), "dst={}, lon1={}, lon2={}, lon={}".format(dst,lon1,lon2,lon)
        assert flge(lat1,lat,lat3), "dst={}, lat1={}, lat3={}, lat={}".format(dst,lat1,lat2,lat)

        # weights
        x, y = lon, lat
        x1, x2 = lon1, lon2
        y1, y2 = lat1, lat3

        return self.get_bilinear_weights(dst, (x,y), (x1,y1), (x2,y2))



    def get_weights_cs2ll(self, dst, alpha, beta, panel, gids):
        cs_obj = self.cs_obj

        (a1,b1), (a2,b2), (a3,b3), (a4,b4) = \
                [cs_obj.alpha_betas[gid] for gid in gids]
        
        assert np.fabs(a1-a3)<1e-15
        assert np.fabs(a2-a4)<1e-15
        assert np.fabs(b1-b2)<1e-15
        assert np.fabs(b3-b4)<1e-15
        assert flge(a1,alpha,a2), "dst={}, a1={}, a2={}, alpha={}".format(dst,a1,a2,alpha)
        assert flge(b1,beta,b3), "dst={}, b1={}, b3={}, beta={}".format(dst,b1,b3,beta)

        panels = [cs_obj.gq_indices[gid,0] for gid in gids]
        for p in panels:
            if p != panel:
                print("(alpha,beta) ({},{})".foramt(alpha, beta))
                print("panel: {}, {}".format(panel, panels))
                print("dst: {}".format(dst))
                print("gids: {}".format(gids))
                sys.exit()

        # weights
        x, y = alpha, beta
        x1, x2 = a1, a2
        y1, y2 = b1, b3

        return self.get_bilinear_weights(dst, (x,y), (x1,y1), (x2,y2))



    def make_remap_matrix_ll2cs(self):
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj
        dst_size = self.dst_size
        mat_size = self.mat_size

        src_address = np.zeros((dst_size, mat_size), 'i4')
        remap_matrix = np.zeros((dst_size, mat_size), 'f8')
        
        for dst in range(dst_size):
            lat, lon = cs_obj.latlons[dst]
            idxs = ll_obj.get_surround_idxs(lat, lon)

            if -1 in idxs:
                src_address[dst,:] = [idxs[0] for i in range(4)]
                remap_matrix[dst,:] = [1,0,0,0]
            else:
                src_address[dst,:] = idxs
                remap_matrix[dst,:] = self.get_weights_ll2cs(dst, lat, lon, idxs)

        return src_address, remap_matrix



    def make_remap_matrix_cs2ll(self):
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj
        dst_size = self.dst_size
        mat_size = self.mat_size

        src_address = np.zeros((dst_size, mat_size), 'i4')
        remap_matrix = np.zeros((dst_size, mat_size), 'f8')
        
        for dst in range(dst_size):
            lat, lon = ll_obj.latlons[dst]
            (alpha, beta, panel), gids = cs_obj.get_surround_4_gids(lat, lon)
            uids = cs_obj.uids[np.array(gids)]

            src_address[dst,:] = uids
            remap_matrix[dst,:] = self.get_weights_cs2ll(dst, alpha, beta, panel, gids)

        return src_address, remap_matrix



    def make_remap_matrix_cs2cs(self):
        src_obj = self.cs_obj
        dst_obj = self.ll_obj
        dst_size = self.dst_size
        mat_size = self.mat_size

        src_address = np.zeros((dst_size, mat_size), 'i4')
        remap_matrix = np.zeros((dst_size, mat_size), 'f8')
        
        for dst in range(dst_size):
            lat, lon = dst_obj.latlons[dst]
            (alpha, beta, panel), gids = src_obj.get_surround_4_gids(lat, lon)
            uids = src_obj.uids[np.array(gids)]

            src_address[dst,:] = uids
            remap_matrix[dst,:] = self.get_weights_cs2ll(dst, alpha, beta, panel, gids)

        return src_address, remap_matrix



    def make_remap_matrix_mpi(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        myrank = comm.Get_rank()

        cs_obj = self.cs_obj
        ll_obj = self.ll_obj
        dst_size = self.dst_size
        mat_size = self.mat_size
        direction = self.direction

        if nproc == 1:
            if direction == 'll2cs':
                return self.make_remap_matrix_ll2cs()
            else:
                return self.make_remap_matrix_cs2ll()

        chunk_size = dst_size//nproc//10
        dsw_dict = dict()   # {dst:{srcs,wgts),...}


        if myrank == 0:
            start = 0
            while start < dst_size:
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send(start, dest=rank, tag=10)
                start += chunk_size

            for i in range(nproc-1):
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send('quit', dest=rank, tag=10)

                slave_dsw_dict = comm.recv(source=rank, tag=20)
                dsw_dict.update(slave_dsw_dict)


            src_address = np.zeros((dst_size, mat_size), 'i4')
            remap_matrix = np.zeros((dst_size, mat_size), 'f8')

            for dst in range(dst_size):
                srcs, wgts = dsw_dict[dst]
                src_address[dst,:] = srcs
                remap_matrix[dst,:] = wgts

            return src_address, remap_matrix


        else:
            while True:
                comm.send(myrank, dest=0, tag=0)
                msg = comm.recv(source=0, tag=10)

                if msg == 'quit':
                    print("Slave rank {} quit.".format(myrank))
                    comm.send(dsw_dict, dest=0, tag=20)

                    return None, None

                start = msg
                end = start + chunk_size
                end = dst_size if end > dst_size else end
                print("rank {}: {} ~ {} ({} %%)".format(myrank, start, end, end/dst_size*100))

                if direction == 'll2cs':
                    for dst in range(start,end):
                        lat, lon = cs_obj.latlons[dst]
                        idxs = ll_obj.get_surround_idxs(lat, lon)

                        if -1 in idxs:
                            srcs = [idxs[0] for i in range(4)]
                            wgts = [1,0,0,0]
                        else:
                            srcs = idxs
                            wgts = self.get_weights_ll2cs(dst, lat, lon, idxs)

                        dsw_dict[dst] = (srcs,wgts)

                else:
                    for dst in range(start,end):
                        lat, lon = ll_obj.latlons[dst]
                        (alpha, beta, panel), gids = cs_obj.get_surround_4_gids(lat, lon)
                        srcs = cs_obj.uids[np.array(gids)]
                        wgts = self.get_weights_cs2ll(dst, alpha, beta, panel, gids)
                        dsw_dict[dst] = (srcs,wgts)



    def set_netcdf_remap_matrix(self, ncf, src_address, remap_matrix):
        ncf.createDimension('dst_size', self.dst_size)
        ncf.createDimension('mat_size', self.mat_size)

        vsrc_address = ncf.createVariable('src_address', 'i4', ('dst_size','mat_size'))
        vremap_matrix = ncf.createVariable('remap_matrix', 'f8', ('dst_size','mat_size'))

        vsrc_address[:] = src_address[:]
        vremap_matrix[:] = remap_matrix[:]
