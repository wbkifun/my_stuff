#------------------------------------------------------------------------------
# filename  : cube_remap.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.18     start
#             2016.1.27     add 'regular-shift_lon' to ll_type option
#
#
# Description: 
#   Remap between cubed-sphere and latlon grid
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
import pygrib
import os
import sys
from math import fsum

import cube_remap_core



class CubeRemap(object):
    def __init__(self, ne, cs_type, nlat, nlon, ll_type, direction, method, remap_matrix_dir):
        self.ne = ne
        self.cs_type = cs_type
        self.nlat = nlat
        self.nlon = nlon
        self.ll_type = ll_type
        self.direction = direction
        self.method = method
        self.remap_matrix_dir = remap_matrix_dir

        assert ne in [30,60,120], 'Wrong argument: ne=%d, only supports one of [30, 60, 120].'%(ne)
        assert cs_type in ['regular','rotated'], "Wrong argument: cs_type=%s, only supports one of ['regular','rotated']"%(cs_type)
        assert ll_type in ['regular','gaussian','include_pole','regular-shift_lon'], "Wrong argument: ll_type=%s, only supports one of ['regular','gaussian','include_pole']"%(ll_type)
        assert direction in ['ll2cs','cs2ll','cs2cs'], "Wrong argument: direction=%s, only supports one of ['ll2cs','cs2ll','cs2cs']"%(direction)
        assert method in ['bilinear','vgecore','lagrange','dominant'], "Wrong argument: method=%s, only supports one of ['bilinear','vgecore','lagrange','dominant']"%(method)


        #---------------------------------------------
        # Read remap matrix
        #---------------------------------------------
        if cs_type == 'regular':
            remap_fpath = '%sne%d/%s/%s_%dx%d_%s.nc'%(remap_matrix_dir,ne,method,direction,nlat,nlon,ll_type)
        elif cs_type == 'rotated':
            remap_fpath = '%sne%d/%s_rotated/%s_%dx%d_%s.nc'%(remap_matrix_dir,ne,method,direction,nlat,nlon,ll_type)

        if os.path.exists(remap_fpath):
            self.ncf = ncf = nc.Dataset(remap_fpath, 'r')
        else:
            print 'A remap matrix file is not found.'
            print remap_fpath
            sys.exit()


        #---------------------------------------------
        # Remapping function
        #---------------------------------------------
        if method in ['vgecore','dominant']:
            self.dst_address = ncf.variables['dst_address'][:]
            self.src_address = ncf.variables['src_address'][:]
            self.remap_matrix = ncf.variables['remap_matrix'][:]
        else:
            mat_size = len( ncf.dimensions['mat_size'] )
            dst_size = len( ncf.dimensions['dst_size'] )
            self.src_address = ncf.variables['src_address'][:].ravel().reshape((mat_size,dst_size), order='F')
            self.remap_matrix = ncf.variables['remap_matrix'][:].ravel().reshape((mat_size,dst_size), order='F')


        self.remap = getattr(self, 'remap_%s'%method)
        self.up_size = ncf.up_size



    def remap_fixed_matrix(self, src_var, dst_var):
        '''
        Bilinear
        Lagrange
        '''
        src_address = self.src_address
        remap_matrix = self.remap_matrix
        ndim = src_var.ndim
        assert ndim in [2,3,4]

        func = getattr(cube_remap_core, 'remap_fixed_%dd'%ndim)

        if ndim == 2:   # (nlat,nlon)
            func(src_address, remap_matrix, src_var.ravel(), dst_var)
        elif ndim == 3:
            nlev, nlat, nlon = src_var.shape
            func(nlat*nlon, nlev, src_address, remap_matrix, src_var.ravel(), dst_var)
        elif ndim == 4:
            time, nlev, nlat, nlon = src_var.shape
            func(nlat*nlon, nlev, time, src_address, remap_matrix, src_var.ravel(), dst_var)


    def remap_bilinear(self, src_var, dst_var):
        self.remap_fixed_matrix(src_var, dst_var)



    def remap_lagrange(self, src_var, dst_var):
        self.remap_fixed_matrix(src_var, dst_var)



    def remap_vgecore(self, src_var, dst_var):
        '''
        V-GECoRe
        '''
        dst_address = self.dst_address
        src_address = self.src_address
        remap_matrix = self.remap_matrix
        assert src_var.ndim in [2,3,4]

        func = getattr(cube_remap_core, 'remap_vgecore_%dd'%src_var.ndim)
        func(dst_address, src_address, remap_matrix, src_var.ravel(), dst_var)



    def remap_dominant(self, src_var, dst_var):
        '''
        Remapping dominant types using V-GECoRe
        '''
        dst_address = self.dst_address
        src_address = self.src_address
        remap_matrix = self.remap_matrix
        assert src_var.ndim == 2
        assert src_var.dtype == np.int32

        func = getattr(cube_remap_core, 'remap_dominant_2d')
        func(dst_address, src_address, remap_matrix, src_var.ravel(), dst_var)



    def create_netcdf(self, fpath):
        try:
            ncf = nc.Dataset(fpath, 'w', format='NETCDF3_CLASSIC')  # for pnetcdf
        except Exception as e:
            print e
            print 'fpath', fpath
            sys.exit()

        ncf.description = 'Remapping between Cubed-sphere and Latlon grid'
        ncf.ne = self.ne
        ncf.cs_type = self.cs_type
        ncf.nlat = self.nlat
        ncf.nlon = self.nlon
        ncf.ll_type = self.ll_type
        ncf.direction = self.direction
        ncf.method = self.method

        return ncf

