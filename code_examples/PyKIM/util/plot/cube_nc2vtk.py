#!/usr/bin/env python

#------------------------------------------------------------------------------
# filename  : cube_nc2vtk.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.19     start
#
# description: 
#   Generate the VTK structured data format on the cubed-sphere
#   By using the visit_writer from VisIt
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
import visit_writer
import argparse


#dir_cs_grid = '/data/khkim/cs_grid/'
from util.grid.path import dir_cs_grid
from util.plot.cube_vtk import CubeVTK2D




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--rotated', action='store_true', help='Korea centered rotation')
parser.add_argument('nc_fpath', type=str, help='path of the NetCDF file')
parser.add_argument('varnames', nargs='*', type=str, help='variable names')
args = parser.parse_args()

rotated = args.rotated
nc_fpath = args.nc_fpath
ncf = nc.Dataset(nc_fpath, 'r')

ncol2ne = {48602:30, 194402:60, 777602:120}

if 'ne' in dir(ncf):
    ne, ngq = ncf.ne, ncf.ngq

elif ncf.dimensions.has_key('ncol'):
    ncol = len( ncf.dimensions['ncol'] )
    ne, ngq = ncol2ne[ncol], 4

elif ncf.dimensions.has_key('ncol_cs'):
    ncol = len( ncf.dimensions['ncol_cs'] )
    ne, ngq = ncol2ne[ncol], 4

output_fpath = './' + nc_fpath.split('/')[-1].replace('.nc', '.vtk')
varname_list = args.varnames
if args.varnames == []:
    varname_list = [str(name) for name in ncf.variables.keys()]

print 'Generate a VTK file from a NetCDF file'
print 'ne=%d, ngq=%d'%(ne, ngq)
print 'target variables:', varname_list
print 'output file:', output_fpath
print ''

cs_vtk = CubeVTK2D(ne, ngq, rotated)
cs_vtk.make_vtk_from_netcdf(output_fpath, ncf, varname_list)
