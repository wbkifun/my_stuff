#------------------------------------------------------------------------------
# filename  : cube_remap_bilinear.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.16    start
#             2016.1.4      add make_remap_matrix_cs2ll()
#                           bugfix negative weight at make_remap_matrix_ll2cs()
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

        if direction == 'll2cs':
            self.src_obj = ll_obj
            self.dst_obj = cs_obj

        elif direction == 'cs2ll':
            self.src_obj = cs_obj
            self.dst_obj = ll_obj

        else:
            print '%s direction is not supported.'%(direction)
            sys.exit()

        self.dst_size = self.dst_obj.nsize
        self.mat_size = 4



    def make_remap_matrix_ll2cs(self):
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj
        dst_size = self.dst_size
        mat_size = self.mat_size

        src_address = np.zeros((dst_size, mat_size), 'i4')
        remap_matrix = np.zeros((dst_size, mat_size), 'f8')
        
        for dst in xrange(dst_size):
            lat, lon = cs_obj.latlons[dst]
            idx1, idx2, idx3, idx4 = ll_obj.get_surround_idxs(lat, lon)

            if -1 in [idx1, idx2, idx3, idx4]:
                src_address[dst,:] = [idx1, idx1, idx1, idx1]
                remap_matrix[dst,:] = [1, 0, 0, 0]

            else:
                lat1, lon1 = ll_obj.latlons[idx1]
                lat2, lon2 = ll_obj.latlons[idx2]
                lat3, lon3 = ll_obj.latlons[idx3]
                lat4, lon4 = ll_obj.latlons[idx4]

                assert np.fabs(lon1-lon3)<1e-15
                assert np.fabs(lon2-lon4)<1e-15
                assert np.fabs(lat1-lat2)<1e-15
                assert np.fabs(lat3-lat4)<1e-15

                if lon2 == 0: lon2 = lon1 + ll_obj.dlon
                if lon4 == 0: lon4 = lon3 + ll_obj.dlon
                assert flge(lon1,lon,lon2), 'dst=%d, lon1=%f, lon2=%f, lon=%f'%(dst,lon1,lon2,lon)
                assert flge(lat1,lat,lat3), 'dst=%d, lat1=%f, lat3=%f, lat=%f'%(dst,lat1,lat2,lat)

                # weights
                x1, x2 = lon1, lon2
                y1, y2 = lat1, lat3
                dx, dy = x2-x1, y2-y1
                dxy = dx*dy
                x, y = lon, lat

                w1 = (x2-x)*(y2-y)/dxy
                w2 = (x-x1)*(y2-y)/dxy
                w3 = (x2-x)*(y-y1)/dxy
                w4 = (x-x1)*(y-y1)/dxy

                w_list = list()
                for w in [w1,w2,w3,w4]: 
                    if np.fabs(w) < 1e-13:
                        w = 0

                    if w < 0:
                        print 'lat,lon', lat, lon
                        print 'dst', dst
                        print 'idxs', idx1, idx2, idx3, idx4
                        print 'x1,x2', x1, x2
                        print 'y1,y2', y1, y2
                        print 'dx,dy', dx, dy
                        print 'x,y', x, y
                        print 'ws', [w1,w2,w3,w4]
                        sys.exit()

                    w_list.append(w)

                src_address[dst,:] = [idx1, idx2, idx3, idx4]
                remap_matrix[dst,:] = w_list

        return src_address, remap_matrix



    def make_remap_matrix_cs2ll(self):
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj
        dst_size = self.dst_size
        mat_size = self.mat_size

        src_address = np.zeros((dst_size, mat_size), 'i4')
        remap_matrix = np.zeros((dst_size, mat_size), 'f8')
        
        for dst in xrange(dst_size):
            lat, lon = ll_obj.latlons[dst]
            abp, gids = cs_obj.get_surround_4_gids(lat, lon)
            uids = cs_obj.uids[np.array(gids)]

            alpha, beta, panel = abp
            (a1,b1), (a2,b2), (a3,b3), (a4,b4) = \
                    [cs_obj.alpha_betas[gid] for gid in gids]
            
            assert np.fabs(a1-a3)<1e-15
            assert np.fabs(a2-a4)<1e-15
            assert np.fabs(b1-b2)<1e-15
            assert np.fabs(b3-b4)<1e-15
            assert flge(a1,alpha,a2), 'dst=%d, a1=%f, a2=%f, alpha=%f'%(dst,a1,a2,alpha)
            assert flge(b1,beta,b3), 'dst=%d, b1=%f, b3=%f, beta=%f'%(dst,b1,b3,beta)

            panels = [cs_obj.gq_indices[gid,0] for gid in gids]
            for p in panels:
                if p != panel:
                    print '(lat, lon)', lat, lon
                    print 'panel', panel, panels
                    print 'dst', dst
                    print 'uids', uids
                    print 'gids', gids
                    sys.exit()

            # weights
            x1, x2 = a1, a2
            y1, y2 = b1, b3
            dx, dy = x2-x1, y2-y1
            dxy = dx*dy
            x, y = alpha, beta

            w1 = (x2-x)*(y2-y)/dxy
            w2 = (x-x1)*(y2-y)/dxy
            w3 = (x2-x)*(y-y1)/dxy
            w4 = (x-x1)*(y-y1)/dxy

            w_list = list()
            for w in [w1,w2,w3,w4]: 
                if np.fabs(w) < 1e-13:
                    w = 0

                if w < 0:
                    print 'lat,lon', lat, lon
                    print 'abp', abp
                    print 'dst', dst
                    print 'uids', uids
                    print 'gids', gids
                    print 'x1,x2', x1, x2
                    print 'y1,y2', y1, y2
                    print 'dx,dy', dx, dy
                    print 'x,y', x, y
                    print 'ws', [w1,w2,w3,w4]
                    sys.exit()

                w_list.append(w)

            src_address[dst,:] = uids
            remap_matrix[dst,:] = w_list

        return src_address, remap_matrix



    def make_remap_matrix(self):
        return getattr(self, 'make_remap_matrix_%s'%self.direction)()



    def make_remap_matrix_mpi(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        myrank = comm.Get_rank()

        if nproc == 1:
            return self.make_remap_matrix()

        else:
            if myrank == 0:
                print 'Error: Not support MPI. It is fast sufficiently with single process.'
            sys.exit()



    def set_netcdf_remap_matrix(self, ncf, src_address, remap_matrix):
        ncf.createDimension('dst_size', self.dst_size)
        ncf.createDimension('mat_size', self.mat_size)

        vsrc_address = ncf.createVariable('src_address', 'i4', ('dst_size','mat_size'))
        vremap_matrix = ncf.createVariable('remap_matrix', 'f8', ('dst_size','mat_size'))

        vsrc_address[:] = src_address[:]
        vremap_matrix[:] = remap_matrix[:]
