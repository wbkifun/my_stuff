#------------------------------------------------------------------------------
# filename  : cube_remap_vgecore.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.16    start
#             2015.12.18    change shapely to inhouse code
#             2015.12.21    insert debug code
#
#
# Description: 
#   Remap between cubed-sphere and latlon grid
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from math import fsum

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal

from util.convert_coord.cart_ll import latlon2xyz
from util.geometry.sphere import intersect_two_polygons, area_polygon




class VGECoRe(object):
    def __init__(self, cs_obj, ll_obj, direction):
        self.cs_obj = cs_obj
        self.ll_obj = ll_obj
        self.direction = direction

        if direction == 'll2cs':
            self.dst_obj = cs_obj
            self.src_obj = ll_obj

        else:
            self.dst_obj = ll_obj
            self.src_obj = cs_obj



    def make_remap_matrix_from_dsw_dict(self, dsw_dict):
        num_links = sum( [len(sw_list) for sw_list in dsw_dict.values()] )
        dst_address = np.zeros(num_links, 'i4')
        src_address = np.zeros(num_links, 'i4')
        remap_matrix = np.zeros(num_links, 'f8')

        seq = 0
        for dst in sorted(dsw_dict.keys()):
            for src, wgt in sorted(dsw_dict[dst]):
                dst_address[seq] = dst
                src_address[seq] = src
                remap_matrix[seq] = wgt
                seq += 1

        return dst_address, src_address, remap_matrix



    def make_remap_matrix(self, debug=False):
        dst_obj = self.dst_obj
        src_obj = self.src_obj

        dsw_dict = dict()       # {dst:[(s,w),(s,w),...],...}
        if debug: ipoly_dict = dict()  # {dst:[(src,ipoly,iarea),...],..}
        
        for dst, (lat0,lon0) in enumerate(dst_obj.latlons):
            #print dst
            dst_poly = dst_obj.get_voronoi(dst)
            dst_area = area_polygon(dst_poly)

            idx1, idx2, idx3, idx4 = src_obj.get_surround_idxs(lat0, lon0)
            candidates = set([idx1, idx2, idx3, idx4])
            if -1 in candidates: candidates.remove(-1)
            checked_srcs = set()

            dsw_dict[dst] = list()
            while len(candidates) > 0:
                #print '\t', checked_srcs
                src = candidates.pop()
                src_poly = src_obj.get_voronoi(src)

                ipoly = intersect_two_polygons(dst_poly, src_poly)
                checked_srcs.add(src)

                if ipoly != None:
                    iarea = area_polygon(ipoly)
                    area_ratio = iarea/dst_area

                    if area_ratio > 1e-10:
                        dsw_dict[dst].append( (src, area_ratio) )
                        nbrs = set(src_obj.get_neighbors(src))
                        candidates.update(nbrs - checked_srcs)

                        if debug: ipoly_dict[dst].append( (src,ipoly,iarea) )

        if debug: self.save_netcdf_ipoly(ipoly_dict)

        return self.make_remap_matrix_from_dsw_dict(dsw_dict)



    def make_remap_matrix_mpi(self, debug=False):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        myrank = comm.Get_rank()

        if nproc == 1:
            return self.make_remap_matrix()


        dst_obj = self.dst_obj
        src_obj = self.src_obj
        chunk_size = dst_obj.nsize//nproc//10

        dsw_dict = dict()       # {dst:[(src,wgt),(src,wgt),...],...}

        if debug:
            ipoly_dict = dict()  # {dst:[(src,ipoly,iarea),...],..}
        

        if myrank == 0:
            start = 0
            while start < dst_obj.nsize:
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send(start, dest=rank, tag=10)
                start += chunk_size

            for i in xrange(nproc-1):
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send('quit', dest=rank, tag=10)

                slave_dsw_dict = comm.recv(source=rank, tag=20)
                dsw_dict.update(slave_dsw_dict)

                if debug:
                    slave_ipoly_dict = comm.recv(source=rank, tag=30)
                    ipoly_dict.update(slave_src_poly_dict)
                    self.save_netcdf_ipoly(ipoly_dict)

            return self.make_remap_matrix_from_dsw_dict(dsw_dict)

        else:
            while True:
                comm.send(myrank, dest=0, tag=0)
                msg = comm.recv(source=0, tag=10)

                if msg == 'quit':
                    print 'Slave rank %d quit.'%(myrank)
                    comm.send(dsw_dict, dest=0, tag=20)

                    if debug:
                        comm.send(ipoly_dict, dest=0, tag=30)

                    return None, None, None

                start = msg
                end = start + chunk_size
                end = dst_obj.nsize if end > dst_obj.nsize else end
                print 'rank %d: %d ~ %d (%d %%)'%(myrank, start, end, end/dst_obj.nsize*100)

                for dst in xrange(start,end):
                    lat0, lon0 = dst_obj.latlons[dst]
                    dst_poly = dst_obj.get_voronoi(dst)
                    dst_area = area_polygon(dst_poly)

                    idx1, idx2, idx3, idx4 = src_obj.get_surround_idxs(lat0, lon0)
                    candidates = set([idx1, idx2, idx3, idx4])
                    if -1 in candidates: candidates.remove(-1)
                    checked_srcs = set()

                    dsw_dict[dst] = list()

                    if debug:
                        ipoly_dict[dst] = list()

                    while len(candidates) > 0:
                        src = candidates.pop()
                        src_poly = src_obj.get_voronoi(src)

                        ipoly = intersect_two_polygons(dst_poly, src_poly)
                        checked_srcs.add(src)

                        if ipoly != None:
                            iarea = area_polygon(ipoly)
                            area_ratio = iarea/dst_area

                            if area_ratio > 1e-10:
                                dsw_dict[dst].append( (src, area_ratio) )
                                nbrs = set(src_obj.get_neighbors(src))
                                candidates.update(nbrs - checked_srcs)

                                if debug:
                                    ipoly_dict[dst].append( (src,ipoly,iarea) )



    def make_netcdf_polygons(self, dst_poly_dict, src_poly_dict):
        '''
        Save Voronoi polygons
        of destination grid, source grid and overlapped polygons
        '''
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj

        #------------------------------------------------------------
        # Get sizes
        #------------------------------------------------------------
        dst_poly_size = sum([len(poly) for poly, area in dst_poly_dict.values()])
        dst_polys = np.zeros((dst_poly_size,3), 'f8')
        dst_areas = np.zeros(cs_obj.up_size, 'f8')

        src_poly_size = 0
        ipoly_size = 0
        iarea_size = 0
        for src, src_poly, ipoly, iarea in dst_poly_dict.values():
            src_poly_size += len(src_poly)
            ipoly_size += len(ipoly)
            iarea_size += 1

        src_polys = np.zeros((src_poly_size,3), 'f8')
        ipolys = np.zeros((ipoly_size,3), 'f8')
        iareas = np.zeros(iarea_size, 'f8')


        ncf = nc.Dataset(fpath, 'w', format='NETCDF3_CLASSIC') # for pnetcdf
        ncf.description = 'Voronoi polygons'
        ncf.remap_direction = 'll2cs'

        ncf.rotated = str(cs_obj.rotated).lower()
        ncf.ne = cs_obj.ne
        ncf.ngq = cs_obj.ngq
        ncf.ep_size = cs_obj.ep_size
        ncf.up_size = cs_obj.up_size
        ncf.nlat = ll_obj.nlat
        ncf.nlon = ll_obj.nlon
        ncf.ll_size = ll_obj.nsize

        ncf.createDimension('num_links', num_links)

        vdst_address = ncf.createVariable('dst_address', 'i4', ('num_links',))
        vsrc_address = ncf.createVariable('src_address', 'i4', ('num_links',))
        vremap_matrix = ncf.createVariable('remap_matrix', 'f8', ('num_links',))

        vdst_address[:] = dst_address[:]
        vsrc_address[:] = src_address[:]
        vremap_matrix[:] = remap_matrix[:]

        ncf.close()



    def set_netcdf_remap_matrix(self, ncf, dst_address, src_address, remap_matrix):
        ncf.createDimension('num_links', dst_address.size)

        vdst_address = ncf.createVariable('dst_address', 'i4', ('num_links',))
        vsrc_address = ncf.createVariable('src_address', 'i4', ('num_links',))
        vremap_matrix = ncf.createVariable('remap_matrix', 'f8', ('num_links',))

        vdst_address[:] = dst_address[:]
        vsrc_address[:] = src_address[:]
        vremap_matrix[:] = remap_matrix[:]
