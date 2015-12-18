#------------------------------------------------------------------------------
# filename  : cube_sparse_matrix.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.9      start
#             2015.9.11     change to class
#             2015.11.30    append SparseMatrixExpand
#
#
# description: 
#   Generate the sparse matrix to exchange the boundary on the cubed sphere
#   1. the spectral element method 
#   2. the implicit diffusion 
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc


fname = __file__.split('/')[-1]
fdir = __file__.rstrip(fname)




class SparseMatrixSE(object):
    '''
    A sparse matrix for the spectral element method
    '''
    def __init__(self, ne, ngq):
        self.ne = ne
        self.ngq = ngq


        #-----------------------------------------------------
        # Read the NetCDF file of the cubed-sphere grid 
        #-----------------------------------------------------
        cs_fpath = fdir + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
        cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
        mvps = cs_ncf.variables['mvps'][:]


        #-----------------------------------------------------
        # Construct the sparse matrix
        #-----------------------------------------------------
        dsts = list()
        srcs = list()
        weights = list()

        for seq, mvp in enumerate(mvps):
            eff_mvp = [k for k in mvp if k != -1]
            Nmvp = len(eff_mvp)

            if Nmvp > 1:
                for m in eff_mvp:
                    dsts.append(seq)
                    srcs.append(m)
                    weights.append( 1./Nmvp )


        #-----------------------------------------------------
        # Global variables
        #-----------------------------------------------------
        self.cs_ncf = cs_ncf
        self.mvps = mvps

        self.spmat_size = len(dsts)
        self.dsts = np.array(dsts, 'i4')
        self.srcs = np.array(srcs, 'i4')
        self.weights = np.array(weights, 'f8')



    def save_netcdf(self):
        ne, ngq = self.ne, self.ngq

        ncf = nc.Dataset('spmat_se_ne%dngq%d.nc'%(ne,ngq), 'w', format='NETCDF4')
        ncf.description = 'Sparse matrix for the spectral element method on the cubed-Sphere'
        ncf.notice = 'All sequential indices start from 0'
        ncf.createDimension('ne', ne)
        ncf.createDimension('ngq', ngq)
        ncf.createDimension('spmat_size', len(self.dsts))

        vdsts = ncf.createVariable('dsts', 'i4', ('spmat_size',))
        vsrcs = ncf.createVariable('srcs', 'i4', ('spmat_size',))
        vweights = ncf.createVariable('weights', 'f8', ('spmat_size',))

        vdsts[:] = self.dsts
        vsrcs[:] = self.srcs
        vweights[:] = self.weights

        ncf.close()




class SparseMatrixExpand(object):
    '''
    A sparse matrix to expand from UP to EP
    '''
    def __init__(self, ne, ngq):
        self.ne = ne
        self.ngq = ngq


        #-----------------------------------------------------
        # Read the NetCDF file of the cubed-sphere grid 
        #-----------------------------------------------------
        cs_fpath = fdir + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
        cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
        mvps = cs_ncf.variables['mvps'][:]
        is_uvps = cs_ncf.variables['is_uvps'][:]


        #-----------------------------------------------------
        # Construct the sparse matrix
        #-----------------------------------------------------
        dsts = list()
        srcs = list()
        weights = list()

        for is_uvp, mvp in zip(is_uvps, mvps):
            if not is_uvp:
                eff_mvp = [k for k in mvp if k != -1]

                for m in eff_mvp[1:]:
                    if is_uvps[m]:
                        dsts.append(mvp[0])
                        srcs.append(m)
                        weights.append(1)


        #-----------------------------------------------------
        # Global variables
        #-----------------------------------------------------
        self.cs_ncf = cs_ncf
        self.mvps = mvps
        self.is_uvps = is_uvps

        self.spmat_size = len(dsts)
        self.dsts = np.array(dsts, 'i4')
        self.srcs = np.array(srcs, 'i4')
        self.weights = np.array(weights, 'f8')



    def save_netcdf(self):
        ne, ngq = self.ne, self.ngq

        ncf = nc.Dataset('spmat_expand_ne%dngq%d.nc'%(ne,ngq), 'w', format='NETCDF4')
        ncf.description = 'Sparse matrix to expand from UP to EP'
        ncf.notice = 'All sequential indices start from 0'
        ncf.createDimension('ne', ne)
        ncf.createDimension('ngq', ngq)
        ncf.createDimension('spmat_size', len(self.dsts))

        vdsts = ncf.createVariable('dsts', 'i4', ('spmat_size',))
        vsrcs = ncf.createVariable('srcs', 'i4', ('spmat_size',))
        vweights = ncf.createVariable('weights', 'f8', ('spmat_size',))

        vdsts[:] = self.dsts
        vsrcs[:] = self.srcs
        vweights[:] = self.weights

        ncf.close()




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('ngq', type=int, help='number of Gauss qudrature points')
    args = parser.parse_args()
    print 'ne=%d, ngq=%d'%(args.ne, args.ngq) 

    spmat_se = SparseMatrixSE(args.ne, args.ngq)
    spmat_se.save_netcdf()
