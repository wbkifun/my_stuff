#!/usr/bin/env python

#------------------------------------------------------------------------------
# filename  : remap_ancillary.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.18     start
#             2016.1.26     add surface_simple()
#             2016.1.27     add check_ll_grid() and transform_ll_grid()
#             2016.1.28     bugfix: dst_var overwritten -> nparray.copy()
#             2016.2.16     change dtype of aerosol_macc float64->float32
#             2016.2.17     split sea_surface_temperature to _gfs and _um
#             2016.3.4      modify sea_surface_temperature() to call from external code
#             2016.5.17     fix aerosol_macc() -> cube_remap_core.f90/*_4d_()
#             2016.5.20     change data: vtype.nc(usgs->igbp), maxsnowalb.nc(high res.)
#             2016.5.24     add ozone_gmao()
#             2016.6.20     modity ozone_gmao(): ozone variable -> ozone01~ozone12
#             2016.6.20     add soil_type_gfs()
#             2016.8.1      add ne240
#
#
# Description: 
#   Remapping Latlon -> Cubed-sphere
#------------------------------------------------------------------------------

import numpy as np
import netCDF4 as nc
import pygrib
import argparse
import os
import sys
from glob import glob
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal

from util.remap.cube_remap import CubeRemap
from util.remap.cube_remap_matrix import make_lats_lons




class RemapAncillary():
    def __init__(self, ne, cs_type, remap_matrix_dir, src_dir, dst_dir):
        self.ngq = ngq = 4
        self.direction = direction = 'll2cs'

        self.ne = ne
        self.cs_type = cs_type
        self.remap_matrix_dir = remap_matrix_dir
        self.src_dir = src_dir
        self.dst_dir = dst_dir


        ne2up_sizes = {30:48602, 60:194402, 120:777602, 240:3110402}
        self.up_size = ne2up_sizes[ne]

        self.base_remap_args = [ne, cs_type, direction, remap_matrix_dir]
        self.remap_objs = dict()     # Avoid redundant object creation



    def create_remap_object(self, remap_args):
        remap_objs = self.remap_objs
        ne, cs_type, direction, remap_matrix_dir = self.base_remap_args

        if remap_args in remap_objs.keys():
            return remap_objs[remap_args]

        else:
            nlat, nlon, ll_type, method = remap_args
            remap_obj = CubeRemap(ne, cs_type, nlat, nlon, ll_type, \
                    direction, method, remap_matrix_dir)
            remap_objs[remap_args] = remap_obj

            return remap_obj



    def check_ll_grid(self, remap_obj, lats, lons, lat_reverse=False, lon_shift=0):
        '''
        lat_reverse : True/False
        lon_shift   : number of shift points (used by np.roll)
        '''
        if lat_reverse: lats = lats[::-1]
        lons = np.roll(lons + (lons<0)*360, lon_shift)

        nlat, nlon = remap_obj.nlat, remap_obj.nlon
        ll_type = remap_obj.ll_type
        ref_lats, ref_lons = make_lats_lons(nlat, nlon, ll_type)

        aa_equal(np.rad2deg(ref_lats), lats, 13)
        aa_equal(np.rad2deg(ref_lons), lons, 13)



    def transform_ll_grid(self, ll_var, lat_reverse=False, lon_shift=0):
        '''
        lat_reverse : True/False
        lon_shift   : number of shift points (used by np.roll)
        '''
        if lat_reverse:
            if   ll_var.ndim == 2: ll_var2 = ll_var[::-1,:]
            elif ll_var.ndim == 3: ll_var2 = ll_var[:,::-1,:]
            elif ll_var.ndim == 4: ll_var2 = ll_var[:,:,::-1,:]
        else:
            ll_var2 = ll_var[:]

        ll_var3 = np.roll(ll_var2, lon_shift, -1)

        return ll_var3



    def surface_simple(self, src_fname, src_vname, dst_vname, ll_type, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        src_dir = self.src_dir
        dst_dir = self.dst_dir

        #-------------------------------------
        # Setup
        #-------------------------------------
        src_fpath = src_dir + src_fname
        dst_fpath = dst_dir + '%s_ne%.3dnp%d.nc'%(dst_vname,ne,ngq)
        dtype = np.float64
        method = 'bilinear'

        if print_on: print("Source: {}".format(src_fpath))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_ncf = nc.Dataset(src_fpath, 'r')
        for key in src_ncf.dimensions.keys():
            if 'lat' in key: lat_name = key
            if 'lon' in key: lon_name = key
        
        nlat = len( src_ncf.dimensions[lat_name] )
        nlon = len( src_ncf.dimensions[lon_name] )

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if ll_type == 'regular':
            lat_reverse = False
            lon_shift = 0
        elif ll_type == 'regular-shift_lon':
            lat_reverse = False
            lon_shift = nlon//2

        if print_on: print("Check latlon grid")
        lats = src_ncf.variables[lat_name][:]
        lons = src_ncf.variables[lon_name][:]
        self.check_ll_grid(remap_obj, lats, lons, 
                lat_reverse=lat_reverse, lon_shift=lon_shift)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(nlat,nlon) -> (up_size,)")
        if print_on: print("({},{}) -> ({},)".format(nlat,nlon,up_size))
        src_var = np.zeros((nlat,nlon), dtype)
        dst_var = np.zeros(up_size, dtype)
        src_var[:] = src_ncf.variables[src_vname][:]
        src_var_transform = self.transform_ll_grid(src_var, \
                lat_reverse=lat_reverse, lon_shift=lon_shift)
        remap_obj.remap(src_var_transform, dst_var)


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('ncol', up_size)    # the name 'ncol' is for PyCube
        vvar = dst_ncf.createVariable(dst_vname, dtype, ('ncol',))
        vvar[:] = dst_var[:]

        dst_ncf.close()



    def max_green_vegetation_fraction(self, print_on=False):
        src_fname = 'maxgfrac_intp_hi2low.nc'
        vname = 'maxgfrac'
        self.surface_simple(src_fname, vname, vname, 'regular', print_on)



    def min_green_vegetation_fraction(self, print_on=False):
        src_fname = 'mingfrac_intp_hi2low.nc'
        vname = 'mingfrac'
        self.surface_simple(src_fname, vname, vname, 'regular', print_on)



    def max_snow_albedo(self, print_on=False):
        #src_fname = 'maxsnoalb_1024x768.nc'
        #vname = 'maxsnoalb'
        #self.surface_simple(src_fname, vname, 'regular', print_on)
        src_fname = 'snoalb.7200x3600.nc'
        self.surface_simple(src_fname, 'vvv', 'maxsnowalb', 'regular-shift_lon', print_on)


    def deep_soil_temperature(self, print_on=False):
        src_fname = 'tg3_1024x768.nc'
        vname = 'tg3'
        self.surface_simple(src_fname, vname, vname, 'regular', print_on)



    def monthly_green_vegetation_fraction(self, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        src_dir = self.src_dir
        dst_dir = self.dst_dir

        #-------------------------------------
        # Setup
        #-------------------------------------
        month = 12
        src_fnames = ['%.2d_gfrac_intp.nc'%(m+1) for m in range(month)]
        src_fpaths = [src_dir+fname for fname in src_fnames]
        dst_fpath = dst_dir + 'gfrac_ne%.3dnp%d.nc'%(ne,ngq)
        src_vname = 'gfrac'
        dst_vname = src_vname
        dtype = np.float64
        ll_type = 'regular'
        method = 'bilinear'

        if print_on: print("Source Directory: {}".format(src_dir))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_ncf = nc.Dataset(src_fpaths[0], 'r')
        nlat = len( src_ncf.dimensions['lat'] )
        nlon = len( src_ncf.dimensions['lon'] )

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lats = src_ncf.variables['lat'][:]
        lons = src_ncf.variables['lon'][:]
        self.check_ll_grid(remap_obj, lats, lons)

        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(nlat,nlon) -> (up_size,)")
        if print_on: print("({},{}) -> ({},)".format(nlat,nlon,up_size))
        src_var = np.zeros((nlat,nlon), dtype)
        dst_var = np.zeros(up_size, dtype)
        dst_var_gather = np.zeros((month,up_size), dtype)
        for m, src_fpath in enumerate(src_fpaths):
            if print_on: print("\t{}, {}".format(src_fnames[m], src_vname))
            src_ncf = nc.Dataset(src_fpath, 'r')
            src_var[:] = src_ncf.variables[src_vname][:]
            remap_obj.remap(src_var, dst_var)
            dst_var_gather[m,:] = dst_var[:]


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('month', month)
        dst_ncf.createDimension('ncol', up_size)

        vvar = dst_ncf.createVariable(dst_vname, dtype, ('month','ncol'))
        vvar[:] = dst_var_gather[:]

        dst_ncf.close()



    def modis(self, src_fname, vnames, print_on):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        src_dir = self.src_dir
        dst_dir = self.dst_dir

        #-------------------------------------
        # Setup
        #-------------------------------------
        src_fpath = src_dir + src_fname
        dst_fname = '%s_ne%.3dnp%d.nc'%(src_fname.rstrip('.nc'), ne, ngq)
        dst_fpath = dst_dir + dst_fname
        dtype = np.float64
        ll_type = 'regular-shift_lon'
        method = 'bilinear'

        if print_on: print("Source: {}".format(src_fpath))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_ncf = nc.Dataset(src_fpath, 'r')
        time = len( src_ncf.dimensions['time'] )
        nlat = len( src_ncf.dimensions['latitude'] )
        nlon = len( src_ncf.dimensions['longitude'] )

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lats = src_ncf.variables['latitude'][:]
        lons = src_ncf.variables['longitude'][:]
        self.check_ll_grid(remap_obj, lats, lons, \
                lat_reverse=True, lon_shift=nlon//2)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(time,nlat,nlon) -> (time,up_size)")
        if print_on: print("({},{},{}) -> ({},{})".format(time,nlat,nlon,time,up_size))
        src_var = np.zeros((time,nlat,nlon), dtype)
        dst_var = np.zeros(time*up_size, dtype)
        dst_vars = list()
        for vname in vnames:
            if print_on: print("\t{}".format(vname))
            src_var[:] = src_ncf.variables[vname][:]

            src_var_transform = self.transform_ll_grid(src_var, \
                    lat_reverse=True, lon_shift=nlon//2)

            remap_obj.remap(src_var_transform, dst_var)
            dst_vars.append( dst_var.reshape(time,up_size).copy() )


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('time', time)
        dst_ncf.createDimension('ncol', up_size)    # the name 'ncol' is for PyCube

        for vname, dst_var in zip(vnames, dst_vars):
            vvar = dst_ncf.createVariable(vname, dtype, ('time','ncol'))
            vvar[:] = dst_var[:]

        dst_ncf.close()



    def modis_albedo(self, print_on=False):
        src_fname = 'MODISalb.nc'
        vnames = ['vis1', 'vis2', 'vis3', 'nir1', 'nir2', 'nir3']
        self.modis(src_fname, vnames, print_on)



    def modis_emissivity(self, print_on=False):
        src_fname = 'MODISems.nc'
        vnames = ['ems']
        self.modis(src_fname, vnames, print_on)



    def ocean_mixed_layer_depth(self, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        src_dir = self.src_dir
        dst_dir = self.dst_dir

        #-------------------------------------
        # Setup
        #-------------------------------------
        src_fpath = src_dir + 'clim.omld.1x1.nc'
        dst_fpath = dst_dir + 'clim_omld_ne%.3dnp%d.nc'%(ne,ngq)
        vname = 'omld'
        dtype = np.float64
        ll_type = 'regular-shift_lon'
        method = 'bilinear'

        if print_on: print("Source: {}".format(src_fpath))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_ncf = nc.Dataset(src_fpath, 'r')
        time = len( src_ncf.dimensions['time'] )
        nlat = len( src_ncf.dimensions['lat'] )
        nlon = len( src_ncf.dimensions['lon'] )

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lats = src_ncf.variables['lat'][:]
        lons = src_ncf.variables['lon'][:]
        self.check_ll_grid(remap_obj, lats, lons)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(time,nlat,nlon) -> (time,up_size)")
        if print_on: print("({},{},{}) -> ({},{})".format(time,nlat,nlon,time,up_size))
        src_var = np.zeros((time,nlat,nlon), dtype)
        dst_var = np.zeros(time*up_size, dtype)

        if print_on: print("\t{}".format(vname))
        src_var[:] = src_ncf.variables[vname][:]
        remap_obj.remap(src_var, dst_var)


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('time', time)
        dst_ncf.createDimension('ncol', up_size)    # the name 'ncol' is for PyCube

        vvar = dst_ncf.createVariable(vname, dtype, ('time','ncol'))
        vvar[:] = dst_var.reshape(time,up_size)[:]

        dst_ncf.close()



    def ice_thickness(self, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        src_dir = self.src_dir
        dst_dir = self.dst_dir

        #-------------------------------------
        # Setup
        #-------------------------------------
        src_fpath = src_dir + 'clim.icetk.grib'
        dst_fpath = dst_dir + 'hice_ne%.3dnp%d.nc'%(ne,ngq)
        src_vname = 'Ice thickness'
        dst_vname = 'hice'
        dtype = np.float64
        ll_type = 'regular-shift_lon'
        method = 'bilinear'

        if print_on: print("Source: {}".format(src_fpath))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_gbf = pygrib.open(src_fpath)
        grbs = src_gbf.select(name=src_vname)
        month = len(grbs)
        nlat, nlon = grbs[0].values.shape

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lats = grbs[0].latlons()[0][:,0]
        lons = grbs[0].latlons()[1][0,:]
        self.check_ll_grid(remap_obj, lats, lons, lat_reverse=True)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(month,nlat,nlon) -> (month,up_size)")
        if print_on: print("({},{},{}) -> ({},{})".format(month,nlat,nlon,month,up_size))
        src_var = np.zeros((nlat,nlon), dtype)
        dst_var = np.zeros(up_size, dtype)
        dst_var_gather = np.zeros((month,up_size), dtype)
        for m, grb in enumerate(grbs):
            if print_on: print("\t{}".format(grb.dataDate))
            src_var[:] = grb.values[:]
            src_var_transform = \
                    self.transform_ll_grid(src_var, lat_reverse=True)

            remap_obj.remap(src_var_transform, dst_var)
            dst_var_gather[m,:] = dst_var[:]


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('month', month)
        dst_ncf.createDimension('ncol', up_size)    # the name 'ncol' is for PyCube

        vvar = dst_ncf.createVariable(dst_vname, dtype, ('month','ncol'))
        vvar[:] = dst_var_gather[:]

        dst_ncf.close()



    def ozone_gmao(self, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        src_dir = self.src_dir
        dst_dir = self.dst_dir

        #-------------------------------------
        # Setup
        #-------------------------------------
        src_fpath = src_dir + 'ozone_gmao.nc'
        dst_fpath = dst_dir + 'clim_ozone_ne{:03d}np{}.nc'.format(ne,ngq)
        vname = 'ozone'
        dtype = np.float64
        ll_type = 'include_pole'
        method = 'bilinear'

        if print_on: print("Source Directory: {}".format(src_dir))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_ncf = nc.Dataset(src_fpath, 'r')
        month = len( src_ncf.dimensions['month'] )
        nlev = len( src_ncf.dimensions['plev'] )
        nlat = len( src_ncf.dimensions['lat'] )
        nlon = len( src_ncf.dimensions['lon'] )

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lats = src_ncf.variables['lat'][:]
        lons = src_ncf.variables['lon'][:]
        self.check_ll_grid(remap_obj, lats, lons, lat_reverse=True)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(month,nlev,nlat,nlon) -> month*(nlev,up_size)")
        if print_on: print("({},{},{},{}) -> {}x({},{})".format(month,nlev,nlat,nlon,month,nlev,up_size))
        src_var = np.zeros((nlev,nlat,nlon), dtype)
        dst_var = np.zeros(nlev*up_size, dtype)
        dst_var_list = list()

        for month in range(12):
            print("month={:02d}".format(month+1))
            src_var[:] = src_ncf.variables[vname][month,:,:,:]
            src_var_transform = self.transform_ll_grid(src_var, lat_reverse=True)
            remap_obj.remap(src_var_transform, dst_var)
            dst_var_list.append(dst_var.copy())


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('nlev', nlev)
        dst_ncf.createDimension('ncol', up_size)

        for month, dst_var in enumerate(dst_var_list):
            vn = "{}{:02d}".format(vname,month+1)
            vvar = dst_ncf.createVariable(vn, dtype, ('nlev','ncol'))
            vvar[:] = dst_var.reshape(nlev,up_size)

        # Copy pressure
        plev = src_ncf.variables['plev']
        vplev = dst_ncf.createVariable('pressure', dtype, ['nlev'])
        vplev.units = plev.units
        vplev[:] = plev[:]

        dst_ncf.close()



    def aerosol_macc(self, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        src_dir = self.src_dir
        dst_dir = self.dst_dir

        #-------------------------------------
        # Setup
        #-------------------------------------
        src_fnames = ['MACC_BC_m.nc', \
                      'MACC_OC_m.nc', \
                      'MACC_SO4_m.nc', \
                      'MACC_SEA_SALT_m.nc', \
                      'MACC_DUST_m.nc']
        src_fpaths = [src_dir+fname for fname in src_fnames]
        dst_fpath = dst_dir + 'clim_aerosol_ne%.3dnp%d.nc'%(ne,ngq)
        vnames = ['BC', 'OC', 'SO4', 'SEASALT', 'DUST']
        dtype = np.float64
        ll_type = 'include_pole'
        method = 'bilinear'

        if print_on: print("Source Directory: {}".format(src_dir))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_ncf = nc.Dataset(src_fpaths[0], 'r')
        month = len( src_ncf.dimensions['month'] )
        nlev = len( src_ncf.dimensions['lev'] )
        nlat = len( src_ncf.dimensions['latitude'] )
        nlon = len( src_ncf.dimensions['longitude'] )

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lat_ncf = nc.Dataset(src_dir+'MACC_lat.nc', 'r')
        lon_ncf = nc.Dataset(src_dir+'MACC_lon.nc', 'r')
        lats = lat_ncf.variables['lat'][:]
        lons = lon_ncf.variables['lon'][:]
        self.check_ll_grid(remap_obj, lats, lons, lat_reverse=True)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(month,nlev,nlat,nlon) -> (month,nlev,up_size)")
        if print_on: print("({},{},{},{}) -> ({},{},{})".format(month,nlev,nlat,nlon,month,nlev,up_size))
        src_var = np.zeros((month,nlev,nlat,nlon), dtype)
        dst_var = np.zeros(month*nlev*up_size, dtype)
        dst_vars = list()
        for i, src_fpath in enumerate(src_fpaths):
            vname = vnames[i]
            if print_on: print("\t{}, {}".format(src_fnames[i], vname))
            src_ncf = nc.Dataset(src_fpath, 'r')
            src_var[:] = src_ncf.variables[vname][:]
            src_var_transform = self.transform_ll_grid(src_var, lat_reverse=True)

            remap_obj.remap(src_var_transform, dst_var)
            dst_vars.append( dst_var.reshape(month,nlev,up_size).copy() )


        # Copy pressure
        src_fname = 'MACC_pres_pa_m.nc'
        vname = 'pressure'
        if print_on: print("\t{}, {}".format(src_fname, vname))
        src_ncf = nc.Dataset(src_dir+src_fname, 'r')
        pres = src_ncf.variables['pressure'][:]


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('month', month)
        dst_ncf.createDimension('nlev', nlev)
        dst_ncf.createDimension('ncol', up_size)

        for vname, dst_var in zip(vnames, dst_vars):
            # np.float32 datatype is used
            vvar = dst_ncf.createVariable(vname, np.float32, ('month','nlev','ncol'))
            print("{}, {}, {}".format(vname, vvar.dtype, vvar.shape))
            vvar[:] = dst_var[:].astype(np.float32)

        vvar = dst_ncf.createVariable('pressure', dtype, ('nlev',))
        vvar[:] = pres[:]

        dst_ncf.close()



    def soil_vegetation_type(self, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        src_dir = self.src_dir
        dst_dir = self.dst_dir

        #-------------------------------------
        # Setup
        #-------------------------------------
        src_fpath = src_dir + 'stype.7200x3600.nc'
        #src_vnames = ['statsgo', 'usgs']
        src_vnames = ['statsgo', 'igbp']
        dst_vnames = ['stype', 'vtype']
        dst_fpaths = [dst_dir+'%s_ne%.3dnp%d.nc'%(t,ne,ngq) for t in dst_vnames]
        dtype = np.int32
        ll_type = 'regular-shift_lon'
        method = 'dominant'

        if print_on: print("Source: {}".format(src_fpath))
        if print_on: print("Destination: {}".format(dst_fpaths))

        src_ncf = nc.Dataset(src_fpath, 'r')
        nlat = len( src_ncf.dimensions['latitude'] )
        nlon = len( src_ncf.dimensions['longitude'] )

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lats = src_ncf.variables['latitude'][:]
        lons = src_ncf.variables['longitude'][:]
        self.check_ll_grid(remap_obj, lats, lons, lat_reverse=True)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(nlat,nlon) -> (up_size,)")
        if print_on: print("({},{}) -> ({},)".format(nlat,nlon,up_size))
        src_var = np.zeros((nlat,nlon), dtype)
        dst_var = np.zeros(up_size, dtype)
        dst_vars = list()
        for vname in src_vnames:
            if print_on: print("\t{}".format(vname))
            src_var[:] = src_ncf.variables[vname][:]
            src_var_transform = self.transform_ll_grid(src_var, lat_reverse=True)

            remap_obj.remap(src_var_transform, dst_var)
            dst_vars.append( dst_var.copy() )


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        for fpath, vname, var in zip(dst_fpaths, dst_vnames, dst_vars):
            dst_ncf = remap_obj.create_netcdf(fpath)
            dst_ncf.createDimension('ncol', up_size)    # the name 'ncol' is for PyCube
            vvar = dst_ncf.createVariable(vname, dtype, ('ncol',))
            vvar[:] = var[:]

        dst_ncf.close()


        #-------------------------------------
        # Create a usgs_roughness file
        #-------------------------------------
        dataname = src_vnames[1]
        if print_on: print("{}_roughness".format(dataname))
        roughness = np.zeros(up_size, np.float64)
        if dataname == 'usgs':
            z0data = [0.50, 0.15, 0.10, 0.15, 0.14, 0.20, 0.12, 0.05, 0.06, 0.15, 0.50, 0.50, 0.50, 0.50, 0.50, 1.59e-5, 0.2, 0.40, 0.01, 0.10, 0.30, 0.15, 0.10, 0.001]
        elif dataname == 'igbp':
            z0data = [0.50, 0.15, 0.10, 0.15, 0.14, 0.20, 0.12, 0.05, 0.06, 0.15, 0.50, 0.50, 0.50, 0.50, 0.50, 1.59e-5, 0.2, 0.40, 0.01, 0.10, 0.30, 0.15, 0.10, 0.001] 

        vtypes = dst_vars[1]
        for idx, vtype in enumerate(vtypes):
            roughness[idx] = z0data[vtype-1]
            fpath = dst_dir+"{}_roughness_ne{:03d}np{}.nc".format(dataname,ne,ngq)

        dst_ncf = remap_obj.create_netcdf(fpath)
        dst_ncf.createDimension('ncol', up_size)    # the name 'ncol' is for PyCube
        vvar = dst_ncf.createVariable('sfcr', np.float64, ('ncol',))
        vvar[:] = roughness[:]

        dst_ncf.close()



    def soil_type_gfs(self, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        src_dir = self.src_dir
        dst_dir = self.dst_dir

        #-------------------------------------
        # Setup
        #-------------------------------------
        src_fpath = src_dir + 'clim.soiltype.grib'
        dst_fpath = dst_dir + 'stype_gfs_ne%.3dnp%d.nc'%(ne,ngq)
        src_vname = 'Snow sublimation heat flux'
        dst_vname = 'stype'
        dtype = np.int32
        ll_type = 'regular-shift_lon'
        method = 'dominant'

        if print_on: print("Source: {}".format(src_fpath))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_gbf = pygrib.open(src_fpath)
        grb = src_gbf.readline()      # only 1 data
        nlat, nlon = grb.values.shape

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lats = grb.latlons()[0][:,0]
        lons = grb.latlons()[1][0,:]
        self.check_ll_grid(remap_obj, lats, lons, lon_shift=nlon//2)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(nlat,nlon) -> (up_size,)")
        if print_on: print("({},{}) -> ({},)".format(nlat,nlon,up_size))
        src_var = np.zeros((nlat,nlon), dtype)
        dst_var = np.zeros(up_size, dtype)

        src_var[:] = grb.values[:]
        src_var_transform = \
                self.transform_ll_grid(src_var, lon_shift=nlon//2)

        remap_obj.remap(src_var_transform, dst_var)


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('ncol', up_size)    # the name 'ncol' is for PyCube

        vvar = dst_ncf.createVariable(dst_vname, dtype, ('ncol',))
        vvar[:] = dst_var[:]

        dst_ncf.close()



    def sea_surface_temperature_gfs(self, kind, date, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        dst_dir = self.dst_dir

        assert kind in ['sst', 'seaice']

        # The date format should be 'yyyymmddhh'
        #date = '20110725'
        #date = '20150820'
        #date = '20160117'
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        src_dir = '/data/DATABASE/ANALYSIS/GFS/gfs_sig/%s/%s/gfs.%s/'%(year,month,date)


        #-------------------------------------
        # Setup
        #-------------------------------------
        src_fpath_list = glob(src_dir + '*%s*'%kind)
        if len(src_fpath_list) == 0:
            if print_on: print("Error: There is no GFS {} file at '{}'.".format(kind, src_dir))
            sys.exit()

        src_fpath = src_fpath_list[0]
        #dst_fpath = dst_dir + 'gfs_sst_%s_ne%.3dnp%d.nc'%(date,ne,ngq)

        if self.cs_type == 'regular':
            dst_base = dst_dir + 'ne%.3dnp%d/%s/%s%s/'%(ne,ngq,kind,year,month)
        elif self.cs_type == 'rotated':
            dst_base = dst_dir + 'ne%.3dnp%d_rotated/%s/%s%s/'%(ne,ngq,kind,year,month)

        if not os.path.exists(dst_base):
            if print_on: print("The destination directory '{}' will be created.".format(dst_base))
            os.mkdir(dst_base)

        dst_fpath = dst_base + 'GFS_%s_%s.nc'%(kind, date)


        if kind == 'sst':
            src_vname = 'Temperature'
            dst_vname = 'sst'
        elif kind == 'seaice':
            src_vname = 'ICEC'
            dst_vname = 'icec'

        dtype = np.float64
        ll_type = 'regular-shift_lon'
        method = 'bilinear'

        if print_on: print("Source: {}".format(src_fpath))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_gbf = pygrib.open(src_fpath)
        grbs = src_gbf.select(name=src_vname)
        nlat, nlon = grbs[0].values.shape

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lats = grbs[0].latlons()[0][:,0]
        lons = grbs[0].latlons()[1][0,:]
        self.check_ll_grid(remap_obj, lats, lons, lat_reverse=True)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(nlat,nlon) -> (up_size,)")
        if print_on: print("({},{}) -> ({})".format(nlat,nlon,up_size))
        src_var = np.zeros((nlat,nlon), dtype)
        dst_var = np.zeros(up_size, dtype)

        grb = grbs[0]
        if print_on: print("\t{}".format(grb.dataDate))
        src_var[:] = grb.values[:]
        src_var_transform = \
                self.transform_ll_grid(src_var, lat_reverse=True)

        remap_obj.remap(src_var_transform, dst_var)


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('ncol', up_size)    # the name 'ncol' is for PyCube

        vvar = dst_ncf.createVariable(dst_vname, dtype, ('ncol',))
        vvar[:] = np.float32(dst_var[:])

        dst_ncf.close()




    def sea_surface_temperature_um(self, kind, src_fpath, print_on=False):
        ne, ngq = self.ne, self.ngq
        up_size = self.up_size
        dst_dir = self.dst_dir

        assert kind in ['sst', 'seaice']

        # The date format should be 'yyyymmdd'
        #date = '20110725'
        #date = '20150820'
        #date = '20151001'
        #date = '20151201'
        #date = '20160117'
        date = src_fpath.split('/')[-1].split('_')[2]
        year = date[:4]
        month = date[4:6]
        day = date[6:8]


        #-------------------------------------
        # Setup
        #-------------------------------------
        if self.cs_type == 'regular':
            dst_base = dst_dir + 'ne%.3dnp%d/%s/%s%s/'%(ne,ngq,kind,year,month)
        elif self.cs_type == 'rotated':
            dst_base = dst_dir + 'ne%.3dnp%d_rotated/%s/%s%s/'%(ne,ngq,kind,year,month)

        if not os.path.exists(dst_base):
            if print_on: print("The destination directory '{}' will be created.".format(dst_base))
            os.mkdir(dst_base)

        dst_fpath = dst_base + 'UM_OSTIA_%s_%s.nc'%(kind,date)


        if kind == 'sst':    vname = 'sst'
        elif kind == 'seaice': vname = 'icec'

        dtype = np.float64
        ll_type = 'regular' 
        method = 'bilinear'

        if print_on: print("Source: {}".format(src_fpath))
        if print_on: print("Destination: {}".format(dst_fpath))

        src_ncf = nc.Dataset(src_fpath, 'r')
        nlat = len( src_ncf.dimensions['lat'] )
        nlon = len( src_ncf.dimensions['lon'] )

        remap_args = (nlat, nlon, ll_type, method)
        remap_obj = self.create_remap_object(remap_args)

        if print_on: print("Check latlon grid")
        lats = src_ncf.variables['lat'][:]
        lons = src_ncf.variables['lon'][:]
        self.check_ll_grid(remap_obj, lats, lons)


        #-------------------------------------
        # Remapping
        #-------------------------------------
        if print_on: print("Remapping using {}".format(method))
        if print_on: print("(nlat,nlon) -> (up_size,)")
        if print_on: print("({},{}) -> ({})".format(nlat,nlon,up_size))
        src_var = np.zeros((nlat,nlon), dtype)
        dst_var = np.zeros(up_size, dtype)

        src_var[:] = src_ncf.variables[vname][:]
        remap_obj.remap(src_var, dst_var)


        #-------------------------------------
        # Save as NetCDF
        #-------------------------------------
        if print_on: print("Save as NetCDF")
        dst_ncf = remap_obj.create_netcdf(dst_fpath)
        dst_ncf.createDimension('ncol', up_size)    # the name 'ncol' is for PyCube

        vvar = dst_ncf.createVariable(vname, dtype, ('ncol',))
        vvar[:] = dst_var[:]

        dst_ncf.close()





if __name__ == '__main__':
    import argparse
    import sys
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    myrank = comm.Get_rank()


    #==========================================================================
    # Setup
    #==========================================================================
    target_ancillaries = [ \
            'max_green_vegetation_fraction', \
            'min_green_vegetation_fraction', \
            'max_snow_albedo', \
            'deep_soil_temperature', \
            'monthly_green_vegetation_fraction', \
            'modis_albedo', \
            'modis_emissivity', \
            'ocean_mixed_layer_depth', \
            'ice_thickness', \
            'aerosol_macc', \
            'soil_vegetation_type', \
            'soil_type_gfs', \
            'ozone_gmao']
    #target_ancillaries = ['sea_surface_temperature_gfs']
    #target_ancillaries = ['sea_surface_temperature_um']
    #==========================================================================


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('cs_type', type=str, help='regular/rotated')
    parser.add_argument('remap_matrix_dir', type=str, help='directory path of remap matrix')
    parser.add_argument('input_dir', type=str, help='directory path of ancillary data')
    parser.add_argument('output_dir', type=str, help='output directory path')
    args = parser.parse_args()

    ne = args.ne
    cs_type = args.cs_type
    remap_matrix_dir = args.remap_matrix_dir
    src_dir = args.input_dir
    dst_dir = args.output_dir

    ancil = RemapAncillary(ne, cs_type, remap_matrix_dir, src_dir, dst_dir)

    if myrank == 0:
        print("{}".format('='*80))
        print("Remapping ancillary data to cubed-sphere grid")
        print("{}".format('='*80))
        print("Target resolution: ne{} ({})".format(ne,cs_type))

        if nproc == 1:
            print("\nSingle process run")
            for ancil_name in target_ancillaries:
                print("\n<{}>".format(ancil_name))
                getattr(ancil, ancil_name)(print_on=True)
            sys.exit()

        for ancil_name in target_ancillaries:
            rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
            comm.send(ancil_name, dest=rank, tag=10)

        for i in xrange(nproc-1):
            rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
            comm.send('quit', dest=rank, tag=10)

    else:
        while True:
            comm.send(myrank, dest=0, tag=0)
            msg = comm.recv(source=0, tag=10)

            if msg == 'quit':
                print("Slave rank {} quit.".format(myrank))
                break
            else:
                print("rank {}: {}".format(myrank, msg))
                getattr(ancil, msg)(print_on=False)
