#------------------------------------------------------------------------------
# filename  : cube_spmat.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.9      start
#             2015.9.11     change to class
#             2015.11.30    append SparseMatrixExpand
#             2016.3.29     convert to Python3
#             2016.8.25     fix the relative import path
#             2016.8.26     modify cs_grid_dpath
#
#
# description: 
#   Generate the sparse matrix to exchange the boundary on the cubed sphere
#   1. Average for the Spectral Element Method 
#   2. Expand from UP to EP 
#   3. The implicit viscosity 
#------------------------------------------------------------------------------

import numpy as np
import netCDF4 as nc
import os

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.extend([current_dpath,dirname(current_dpath)])
from path import cs_grid_dpath




class SparseMatrixAvg:
    '''
    A sparse matrix to average for the spectral element method
    '''
    def __init__(self, ne, ngq):
        self.ne = ne
        self.ngq = ngq


        #-----------------------------------------------------
        # Read the NetCDF file of the cubed-sphere grid 
        #-----------------------------------------------------
        cs_fpath = cs_grid_dpath + "cs_grid_ne{:03d}np{}.nc".format(ne, ngq)
        assert os.path.exists(cs_fpath), "{} is not found.".format(cs_fpath)
        cs_ncf = nc.Dataset(cs_fpath, 'r')
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



    def save_netcdf(self, output_dir):
        ne, ngq = self.ne, self.ngq

        fpath = output_dir + "spmat_avg_ne{:03d}np{}.nc".format(ne,ngq)
        ncf = nc.Dataset(fpath, 'w')
        ncf.description = 'Sparse matrix for the spectral element method on the cubed-Sphere'
        ncf.notice = 'All sequential indices start from 0'
        ncf.method = 'average'

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




class SparseMatrixExpand:
    '''
    A sparse matrix to expand from UP to EP
    '''
    def __init__(self, ne, ngq):
        self.ne = ne
        self.ngq = ngq


        #-----------------------------------------------------
        # Read the NetCDF file of the cubed-sphere grid 
        #-----------------------------------------------------
        cs_fpath = cs_grid_dpath + "cs_grid_ne{:03d}np{}.nc".format(ne, ngq)
        assert os.path.exists(cs_fpath), "{} is not found.".format(cs_fpath)
        cs_ncf = nc.Dataset(cs_fpath, 'r')
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



    def save_netcdf(self, output_dir):
        ne, ngq = self.ne, self.ngq

        fpath = output_dir + "spmat_expand_ne{:03d}np{}.nc".format(ne,ngq)
        ncf = nc.Dataset(fpath, 'w')

        ncf.description = 'Sparse matrix to expand from UP to EP'
        ncf.notice = 'All sequential indices start from 0'
        ncf.method = 'expand'

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
    import sys

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('ngq', type=int, help='number of Gauss qudrature points')
    parser.add_argument('method', type=str, help='spmat method', \
            choices=['average', 'expand'])
    parser.add_argument('output_dir', nargs='?', type=str, default='./', help='output directory')
    args = parser.parse_args()

    print("Generate the information of Cubed-sphere grid")
    print("ne={}, ngq={}".format(args.ne, args.ngq))
    print("method: {}".format(args.method))
    print("output directory: {}".format(args.output_dir))

    yn = raw_input('Continue (Y/n)? ')
    if yn.lower() == 'n': sys.exit()

    
    if args.method == 'average':
        spmat = SparseMatrixAvg(args.ne, args.ngq)

    elif args.method == 'expand':
        spmat = SparseMatrixExpand(args.ne, args.ngq)

    spmat.save_netcdf(args.output_dir)
