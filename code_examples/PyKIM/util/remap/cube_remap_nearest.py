#------------------------------------------------------------------------------
# filename  : cube_remap_nearest.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.8.10     start
#
#
# Description: 
#   Remap between cubed-sphere and other grid
#------------------------------------------------------------------------------

import numpy as np
from math import fsum

from util.convert_coord.cart_ll import latlon2xyz
from util.geometry.sphere import intersect_two_polygons, area_polygon, pt_in_polygon




class Nearest(object):
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

        self.dst_size = self.dst_obj.nsize



    def make_remap_matrix(self, debug=False):
        dst_obj = self.dst_obj
        src_obj = self.src_obj
        dst_size = self.dst_size

        src_address = np.zeros(dst_size, 'i4')

        for dst, (lat0,lon0) in enumerate(dst_obj.latlons):
            uid = src_obj.get_nearest_idx(lat0, lon0)
            src_address[dst] = uid

        return src_address



    def make_remap_matrix_mpi(self, debug=False):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        myrank = comm.Get_rank()

        if nproc == 1:
            return self.make_remap_matrix()


        dst_obj = self.dst_obj
        src_obj = self.src_obj
        dst_size = self.dst_size
        chunk_size = dst_size//nproc//10

        ds_dict = dict()       # {dst:src, ...}

        if myrank == 0:
            start = 0
            while start < dst_nsize:
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send(start, dest=rank, tag=10)
                start += chunk_size

            for i in range(nproc-1):
                rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
                comm.send('quit', dest=rank, tag=10)

                slave_ds_dict = comm.recv(source=rank, tag=20)
                ds_dict.update(slave_ds_dict)

            src_address = np.zeros(dst_size, 'i4')
            for dst in range(dst_size):
                src_address[dst] = ds_dict[dst]

            return src_address

        else:
            while True:
                comm.send(myrank, dest=0, tag=0)
                msg = comm.recv(source=0, tag=10)

                if msg == 'quit':
                    print("Slave rank {} quit.".format(myrank))
                    comm.send(ds_dict, dest=0, tag=20)

                    return None, None

                start = msg
                end = start + chunk_size
                end = dst_obj.nsize if end > dst_obj.nsize else end
                print("rank {}: {} ~ {} ({} %%)".format(myrank, start, end, end/dst_size*100))

                for dst in range(start,end):
                    lat0, lon0 = dst_obj.latlons[dst]
                    uid = src_obj.get_nearest_idx(lat0, lon0)
                    ds_dict[dst] = uid



    def set_netcdf_remap_matrix(self, ncf, src_address):
        ncf.createDimension('dst_size', self.dst_size)

        vsrc_address = ncf.createVariable('src_address', 'i4', ('dst_size',))
        vsrc_address[:] = src_address[:]
