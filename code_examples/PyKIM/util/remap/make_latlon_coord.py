#!/usr/bin/env python

#------------------------------------------------------------------------------
# filename  : make_latlon_coord.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.3.11     start
#
#
# Description: 
#   Generate coordinates of a latlon grid
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import argparse
import netCDF4 as nc

from util.remap.cube_remap_matrix import make_lats_lons 




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('nlon', type=int, help='number of longitude grid')
parser.add_argument('nlat', type=int, help='number of latitude grid')
parser.add_argument('ll_type', type=str, help='latlon grid type')
args = parser.parse_args()


nlat, nlon = args.nlat, args.nlon
lats, lons = make_lats_lons(nlat, nlon, args.ll_type)
lats_deg = np.rad2deg(lats)
lons_deg = np.rad2deg(lons)


fname = 'll_coord_%dx%d.nc'%(nlon, nlat)
ncf = nc.Dataset(fname, 'w', format='NETCDF3_CLASSIC') 
ncf.description = 'Latlon grid coordinates'
ncf.ll_type = args.ll_type

ncf.createDimension('nlon', nlon)
ncf.createDimension('nlat', nlat)

vlons = ncf.createVariable('lons', 'f8', ('nlon',))
vlons.units = 'degrees'
vlats = ncf.createVariable('lats', 'f8', ('nlat',))
vlats.units = 'degrees'

vlons[:] = lons_deg
vlats[:] = lats_deg

ncf.close()
