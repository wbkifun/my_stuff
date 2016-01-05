#------------------------------------------------------------------------------
# filename  : cube_remap_rbf.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.23    start
#
#
# Description: 
#   Remap between cubed-sphere and latlon grid
#   using the Radial basis function
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import sys
from math import sqrt, pi, fsum

from util.geometry.sphere import angle
from util.convert_coord.cart_ll import latlon2xyz




class RadialBasisFunction(object):
    def __init__(self, cs_obj, ll_obj, direction, radius_level):
        self.cs_obj = cs_obj
        self.ll_obj = ll_obj
        self.direction = direction
        self.radius_level = radius_level    # start from 1

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
        self.mat_size = (radius_level*2)**2



    def get_r0(self):
        '''
        Define r0 with average distance between nodes
        '''
        src_obj = self.src_obj
        direction = self.direction
        stage = self.radius_level

        if direction == 'll2cs':
            idxs = set( src_obj.get_surround_idxs(lat=pi/4, lon=pi) )

        elif direction == 'cs2ll':
            idxs = set( src_obj.get_surround_idxs(lat=pi/8, lon=pi/8) )

        for i in xrange(stage-1):
            for idx in idxs.copy():
                idxs.update( src_obj.get_neighbors(idx) )

        distances = list()
        for idx1 in idxs:
            for idx2 in idxs:
                if idx1 != idx2:
                    xyz1 = src_obj.xyzs[idx1]
                    xyz2 = src_obj.xyzs[idx2]
                    distances.append( angle(xyz1, xyz2) )

        return np.average(distances)



    def get_src_address(self, dst):
        src_obj = self.src_obj
        dst_obj = self.dst_obj
        stage = self.radius_level
        mat_size = self.mat_size

        idxs = set( src_obj.get_surround_idxs(*dst_obj.latlons[dst]) )
        for i in xrange(stage-1):
            for idx in idxs.copy():
                idxs.update( src_obj.get_neighbors(idx) )

        if len(idxs) != mat_size:
            print 'dst=%d : len(idxs)=%d is not same with mat_size=%d'%(dst, len(idxs), mat_size)

            if len(idxs) < mat_size:
                for idx in idxs.copy():
                    idxs.update( src_obj.get_neighbors(idx) )

        return sorted(idxs)[:mat_size]     # list



    def get_inverse_matrix(self, r0, dst, srcs):
        '''
        Inverse matrix to solve linear equations
        '''
        src_obj = self.src_obj
        dst_obj = self.dst_obj
        mat_size = self.mat_size

        amat = np.zeros((mat_size, mat_size), 'f8')
        ll_local_xyzs = src_obj.xyzs[srcs]
        func = lambda r: sqrt(r*r + r0*r0)    # multiquadratic

        for i in xrange(mat_size):
            for j in xrange(mat_size):
                r = angle(ll_local_xyzs[i], ll_local_xyzs[j])
                amat[i,j] = func(r)

        invmat = np.linalg.inv(amat)

        for i, src in enumerate(srcs):
            r = angle(dst_obj.xyzs[dst], src_obj.xyzs[src])
            invmat[i,:] *= func(r)

        return invmat



    def make_remap_matrix(self):
        src_obj = self.src_obj
        dst_obj = self.dst_obj
        mat_size = self.mat_size
        dst_size = self.dst_size

        src_address = np.zeros((dst_size, mat_size), 'i4')
        remap_matrix = np.zeros((dst_size, mat_size, mat_size), 'f8')
        r0 = self.r0 = self.get_r0()

        for dst in xrange(dst_size):
            srcs = self.get_src_address(dst)
            invmat = self.get_inverse_matrix(r0, dst, srcs)

            src_address[dst,:] = srcs
            remap_matrix[dst,:,:] = invmat

        return src_address, remap_matrix



    def make_remap_matrix_mpi(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        myrank = comm.Get_rank()

        if nproc == 1:
            return self.make_remap_matrix()

        src_obj = self.src_obj
        dst_obj = self.dst_obj

        dst_size = self.dst_size
        chunk_size = dst_size//nproc//10
        r0 = self.r0 = self.get_r0()
        dsw_dict = dict()   # {dst:(srcs,invmat),...}


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


            mat_size = self.mat_size
            src_address = np.zeros((dst_size, mat_size), 'i4')
            remap_matrix = np.zeros((dst_size, mat_size, mat_size), 'f8')

            for dst in xrange(dst_size):
                srcs, invmat = dsw_dict[dst]
                src_address[dst,:] = srcs
                remap_matrix[dst,:,:] = invmat

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
                    srcs = self.get_src_address(dst)
                    invmat = self.get_inverse_matrix(r0, dst, srcs)
                    dsw_dict[dst] = (srcs,invmat)



    def remapping(self, src_f, dst_f):
        src_obj = self.src_obj
        dst_obj = self.dst_obj
        dst_size = self.dst_size
        mat_size = self.mat_size

        print 'Generate a remap matrix...'
        src_address, remap_matrix = self.make_remap_matrix()

        print 'Remapping...'
        for dst in xrange(dst_size):
            srcs = src_address[dst,:]
            invmat = remap_matrix[dst,:,:]

            wgts = np.dot(invmat, src_f[srcs])
            dst_f[dst] = fsum(wgts)



    def set_netcdf_remap_matrix(self, ncf, src_address, remap_matrix):
        ncf.radius_level = self.radius_level 
        ncf.r0 = self.r0

        ncf.createDimension('dst_size', self.dst_size)
        ncf.createDimension('mat_size', self.mat_size)

        vsrc_address = ncf.createVariable('src_address', 'i4', ('dst_size','mat_size'))
        vremap_matrix = ncf.createVariable('remap_matrix', 'f8', ('dst_size','mat_size','mat_size'))

        vsrc_address[:] = src_address[:]
        vremap_matrix[:] = remap_matrix[:]
