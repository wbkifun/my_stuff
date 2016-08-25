#------------------------------------------------------------------------------
# filename  : cube_remap_matrix_cs2cs.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.3.25     start, split from cube_remap_matrix.py
#
#
# Description: 
#   Generate a remap_matrix to remap from cubed-sphere to rotated cubed-sphere
#------------------------------------------------------------------------------

import numpy as np
import netCDF4 as nc
import argparse
import os
import sys
from mpi4py import MPI

from cube_remap_matrix import CubeGridRemap



comm = MPI.COMM_WORLD
nproc = comm.Get_size()
myrank = comm.Get_rank()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('src_ne', type=int, help='number of elements')
parser.add_argument('dst_ne', type=int, help='number of elements')
parser.add_argument('src_type', type=str, help='cube grid type', \
        choices=['regular', 'rotated'])
parser.add_argument('dst_type', type=str, help='cube grid type', \
        choices=['regular', 'rotated'])
parser.add_argument('method', type=str, help='remap method', \
        choices=['bilinear', 'vgecore', 'rbf', 'lagrange'])
parser.add_argument('output_dir', nargs='?', type=str, \
        help='output directory', default='./remap_matrix/')
args = parser.parse_args()


ngq = 4
src_ne = args.src_ne
dst_ne = args.dst_ne
src_type = args.src_type
dst_type = args.dst_type
method = args.method
output_dir = args.output_dir
output_fname = "remap_cs2cs_ne{}_ne{}_{}_{}_{}.nc".format(src_ne, dst_ne, src_type, dst_type, method)
output_fpath = output_dir + output_fname


#-------------------------------------------------
# check
#-------------------------------------------------
if myrank == 0:
    print("source grid: cubed-sphere({}), ne={}".format(src_type, src_ne))
    print("target grid: cubed-sphere({}), ne={}".format(dst_type, dst_ne))
    print("remap method: {}".format(method))
    print("output directory: {}".format(output_dir))
    print("output filename: {}".format(output_fname))

    #yn = raw_input('Continue (Y/n)? ')
    #if yn.lower() == 'n':
    #    sys.exit()

    if not os.path.exists(output_dir):
        print("{} is not found. Make output directory.".format(output_dir))
        os.makedirs(output_dir)

    #if os.path.exists(output_fpath):
    #    yn = raw_input("%s is found. Overwrite(Y/n)? "%output_fpath)
    #    if yn.lower() == 'n':
    #        sys.exit()

comm.Barrier()


if myrank ==0:
    #------------------------------------------------------------
    print("Prepare to save as NetCDF")
    #------------------------------------------------------------
    ncf = nc.Dataset(output_fpath, 'w', format='NETCDF3_CLASSIC')
    ncf.description = 'Remapping between Cubed-sphere and Latlon grids'
    ncf.remap_method = method


#-------------------------------------------------
if myrank == 0: print("Make a remap matrix")
#-------------------------------------------------
src_obj = CubeGridRemap(src_ne, ngq, src_type=='rotated')
dst_obj = CubeGridRemap(dst_ne, ngq, dst_type=='rotated')

assert method in ['bilinear'], "The remap method {} is not supported yet.".format(method)

if method == 'bilinear':
    from cube_remap_bilinear import Bilinear
    rmp = Bilinear(src_obj, dst_obj, 'cs2cs')
    src_address, remap_matrix = rmp.make_remap_matrix_mpi()

    '''
elif method == 'vgecore':
    from cube_remap_vgecore import VGECoRe
    rmp = VGECoRe(cs_obj, ll_obj, direction)
    dst_address, src_address, remap_matrix = rmp.make_remap_matrix_mpi()

elif method == 'rbf':
    from cube_remap_rbf import RadialBasisFunction
    rmp = RadialBasisFunction(cs_obj, ll_obj, direction)
    src_address, remap_matrix = rmp.make_remap_matrix_mpi()

elif method == 'lagrange':
    assert direction=='cs2ll', "Lagrange method supports only 'cs2ll'"
    from cube_remap_lagrange import LagrangeBasisFunction
    rmp = LagrangeBasisFunction(cs_obj, ll_obj)
    src_address, remap_matrix = rmp.make_remap_matrix_mpi()
    '''


if myrank ==0:
    #------------------------------------------------------------
    print("Save as NetCDF")
    #------------------------------------------------------------
    ncf.ngq = ngq
    ncf.src_ne = src_ne
    ncf.dst_ne = dst_ne
    ncf.src_type = src_type
    ncf.dst_type = dst_type

    if method == 'vgecore':
        rmp.set_netcdf_remap_matrix(ncf, dst_address, src_address, remap_matrix)
    else:
        rmp.set_netcdf_remap_matrix(ncf, src_address, remap_matrix)

    ncf.close()
    print("Done.")
