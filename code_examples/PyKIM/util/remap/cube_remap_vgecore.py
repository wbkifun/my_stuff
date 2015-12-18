#------------------------------------------------------------------------------
# filename  : cube_remap_vgecore.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.16    start
#
#
# Description: 
#   Remap between cubed-sphere and latlon grid
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from shapely.geometry import Polygon
from math import fsum

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal

from pkg.convert_coord.cs_ll import latlon2xyp
from pkg.convert_coord.cart_rotate import xyz_rotate_reverse
from area_sphere import area_polygon_sphere




class VGECoRe(object):
    def __init__(self, cs_obj, ll_obj):
        self.cs_obj = cs_obj
        self.ll_obj = ll_obj



    def make_dsw_dict_ll2cs(self):
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj

        dsw_dict = dict()       # {dst:[(s,w),(s,w),...],...}
        
        for dst, (rlat,rlon) in enumerate(cs_obj.latlons):
            #print dst
            dst_xy_vertices, vor_obj = cs_obj.get_voronoi(dst)
            dst_poly = Polygon(dst_xy_vertices)

            idx1, idx2, idx3, idx4 = ll_obj.get_surround_idxs(rlat, rlon)
            candidates = set([idx1, idx2, idx3, idx4])
            if -1 in candidates: candidates.remove(-1)
            used_srcs = set()

            dsw_dict[dst] = list()
            while len(candidates) > 0:
                #print '\t', used_srcs
                src = candidates.pop()
                ll_vertices = ll_obj.get_voronoi(src)
                src_xy_vertices = [latlon2xyp(lat,lon,rlat,rlon)[1] \
                                   for lat,lon in ll_vertices]
                src_poly = Polygon(src_xy_vertices)

                ipoly = dst_poly.intersection(src_poly)
                area_ratio = ipoly.area/dst_poly.area

                if area_ratio > 1e-10:
                    dsw_dict[dst].append( (src, area_ratio) )
                    used_srcs.add(src)
                    nbrs = set(ll_obj.get_neighbors(src))
                    candidates.update(nbrs - used_srcs)

        return dsw_dict



    def make_dsw_dict_ll2cs_mpi(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        myrank = comm.Get_rank()

        cs_obj = self.cs_obj
        ll_obj = self.ll_obj
        chunk_size = cs_obj.up_size//nproc//10

        dsw_dict = dict()       # {dst:[(s,w),(s,w),...],...}
        

        if myrank == 0:
            start = 0
            while start < cs_obj.up_size:
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send(start, dest=rank, tag=10)
                start += chunk_size

            for i in xrange(nproc-1):
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send('quit', dest=rank, tag=10)
                slave_dsw_dict = comm.recv(source=rank, tag=20)
                dsw_dict.update(slave_dsw_dict)

            return dsw_dict

        else:
            while True:
                comm.send(myrank, dest=0, tag=0)
                msg = comm.recv(source=0, tag=10)

                if msg == 'quit':
                    print 'Slave rank %d quit.'%(myrank)
                    comm.send(dsw_dict, dest=0, tag=20)
                    return dsw_dict

                start = msg
                end = start + chunk_size
                end = cs_obj.up_size if end > cs_obj.up_size else end
                print 'rank %d: %d ~ %d'%(myrank, start, end)

                for dst in xrange(start,end):
                    rlat, rlon = cs_obj.latlons[dst]
                    dst_xy_vertices, vor_obj = cs_obj.get_voronoi(dst)
                    dst_poly = Polygon(dst_xy_vertices)

                    idx1, idx2, idx3, idx4 = ll_obj.get_surround_idxs(rlat, rlon)
                    candidates = set([idx1, idx2, idx3, idx4])
                    if -1 in candidates: candidates.remove(-1)
                    used_srcs = set()

                    dsw_dict[dst] = list()
                    while len(candidates) > 0:
                        src = candidates.pop()
                        ll_vertices = ll_obj.get_voronoi(src)
                        src_xy_vertices = [latlon2xyp(lat,lon,rlat,rlon)[1] \
                                           for lat,lon in ll_vertices]
                        src_poly = Polygon(src_xy_vertices)

                        ipoly = dst_poly.intersection(src_poly)
                        area_ratio = ipoly.area/dst_poly.area

                        if area_ratio > 1e-10:
                            dsw_dict[dst].append( (src, area_ratio) )
                            used_srcs.add(src)
                            nbrs = set(ll_obj.get_neighbors(src))
                            candidates.update(nbrs - used_srcs)
