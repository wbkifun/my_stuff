#!/usr/bin/env python

#------------------------------------------------------------------------------
# filename  : remap_topo_cesm.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.3.25     start
#
#
# Description: 
#   Remapping Cubed-Sphere -> Cubed-sphere to make a rotated CESM topo
#------------------------------------------------------------------------------

import numpy as np
import netCDF4 as nc
import argparse

from cube_remap_core import remap_fixed_2d




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('src_fpath', type=str, help='path of the source topo file')
args = parser.parse_args()


#------------------------------------------------------------------
# CESM topo file
#------------------------------------------------------------------
src_ncf = nc.Dataset(args.src_fpath, 'r')
src_phis = src_ncf.variables['PHIS'][:]
src_sgh = src_ncf.variables['SGH'][:]
ne = src_ncf.ne


#------------------------------------------------------------------
# Remap matrix
#------------------------------------------------------------------
remap_matrix_dir = "/data/KIM2.3/remap_matrix/ne{:03d}np4_rotated/bilinear".format(ne)
remap_fpath = "{}/cs2cs_ne{:03d}np4_regular.nc".format(remap_matrix_dir, ne)

remap_ncf = nc.Dataset(remap_fpath, 'r')
mat_size = len( remap_ncf.dimensions['mat_size'] )
dst_size = len( remap_ncf.dimensions['dst_size'] )
src_address = remap_ncf.variables['src_address'][:].ravel().reshape((mat_size,dst_size), order='F')
print(src_address.shape, src_address.dtype)
remap_matrix = remap_ncf.variables['remap_matrix'][:].ravel().reshape((mat_size,dst_size), order='F')
print(remap_matrix.shape, remap_matrix.dtype)


#------------------------------------------------------------------
# Rotated topo 
#------------------------------------------------------------------
dst_phis = np.zeros(src_phis.size, src_phis.dtype)
dst_sgh = np.zeros(src_sgh.size, src_sgh.dtype)
remap_fixed_2d(src_address, remap_matrix, src_phis, dst_phis)
remap_fixed_2d(src_address, remap_matrix, src_sgh, dst_sgh)


#------------------------------------------------------------------
# Save as netcdf
#------------------------------------------------------------------
out_fpath = "./topo_cesm_ne{:03d}np4_rotated.nc".format(ne)
out_ncf = nc.Dataset(out_fpath, 'w', format='NETCDF3_64BIT')  # for pnetcdf

out_ncf.description = "Topography on the rotated cubed-sphere grid from CESM"
out_ncf.source = args.src_fpath
out_ncf.np = 4
out_ncf.ne = ne
out_ncf.rotated = "true"

out_ncf.createDimension('ncol', dst_size)

vphis = out_ncf.createVariable('PHIS', dst_phis.dtype, ('ncol',))
vphis[:] = dst_phis[:]

vsgh = out_ncf.createVariable('SGH', dst_sgh.dtype, ('ncol',))
vsgh[:] = dst_sgh[:]

out_ncf.close()
