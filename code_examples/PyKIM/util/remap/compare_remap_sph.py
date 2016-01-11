#------------------------------------------------------------------------------
# filename  : remap_sph.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.4   start
#
#
# Description: 
#   Remapping experiements with Spherical harmonics
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
import sys
import os
from math import fsum
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

from cube_remap import CubeGridRemap, LatlonGridRemap
from scipy.special import sph_harm
from util.misc.standard_errors import sem_1_2_inf
from util.plot.cube_vtk import CubeVTK2D
from util.plot.latlon_vtk import LatlonVTK2D




#----------------------------------------------------------
# Setup
#----------------------------------------------------------
method = 'rbf'          # 'bilinear', 'vgecore', 'rbf', 'lagrange', 'vgecore_old', 'scrip'
direction = 'cs2ll'
cs_type = 'rotated'     # 'regular', 'rotated'
ll_type = 'regular'     # 'regular', 'gaussian'

#ne, ngq = 15, 4
ne, ngq = 30, 4
#ne, ngq = 60, 4
#ne, ngq = 120, 4

#nlat, nlon = 90, 180
nlat, nlon = 180, 360
#nlat, nlon = 360, 720
#nlat, nlon = 720, 1440

#nlat, nlon = 192, 384


#----------------------------------------------------------
rotated = cs_type=='rotated'
cs_obj = CubeGridRemap(ne, ngq, rotated)
ll_obj = LatlonGridRemap(nlat, nlon, ll_type)


if direction == 'll2cs':
    src_obj, dst_obj = ll_obj, cs_obj
else:
    src_obj, dst_obj = cs_obj, ll_obj

SCRIP = method in ['vgecore_old', 'scrip']


#
# Test Function
#
m, n = 16, 32
testfunc = lambda lat,lon: sph_harm(m, n, lon, np.pi/2-lat).real

src_f = np.zeros(src_obj.nsize, 'f8')
for i, (lat,lon) in enumerate(src_obj.latlons):
    src_f[i] = testfunc(lat,lon)


#
# Print Setup
#
print 'ne=%d, ngq=%d, %s'%(ne, ngq, cs_type)
print 'nlat=%d, nlon=%d, %s'%(nlat, nlon, ll_type)
print 'method: %s'%(method)
print 'direction: %s'%(direction)
print 'SPH m=%d, n=%d'%(m, n)
print 'SCRIP format: %s'%(SCRIP)


#----------------------------------------------------------
# Remap
#----------------------------------------------------------
dst_f = np.zeros(dst_obj.nsize, src_f.dtype)


remap_dir = '/nas2/user/khkim/remap_matrix/'
fname = 'remap_%s_ne%d_%s_%dx%d_%s_%s.nc'%(direction, ne, cs_type, nlat, nlon, ll_type, method)
fpath = remap_dir + fname
if not os.path.exists(fpath):
    print '%s not found'%fpath
    sys.exit()

ncf = nc.Dataset(fpath, 'r', 'NETCDF3_CLASSIC')


if method in ['bilinear', 'lagrange']:
    dst_size = len( ncf.dimensions['dst_size'] )
    src_address = ncf.variables['src_address'][:]
    remap_matrix = ncf.variables['remap_matrix'][:]

    for dst in xrange(dst_size):
        srcs = src_address[dst,:]
        wgts = remap_matrix[dst,:]
        dst_f[dst] = fsum( src_f[srcs]*wgts )


elif method in ['vgecore', 'vgecore_old', 'scrip']:
    num_links = len( ncf.dimensions['num_links'] )

    if SCRIP:
        dsts = ncf.variables['dst_address'][:] - 1
        srcs = ncf.variables['src_address'][:] - 1
        wgts = ncf.variables['remap_matrix'][:][:,0]
    else:
        dsts = ncf.variables['dst_address'][:]
        srcs = ncf.variables['src_address'][:]
        wgts = ncf.variables['remap_matrix'][:]

    for dst, src, wgt in zip(dsts, srcs, wgts):
        dst_f[dst] += src_f[src]*wgt


elif method == 'rbf':
    '''
    from cube_remap_rbf import RadialBasisFunction

    rbf = RadialBasisFunction(cs_obj, ll_obj, direction, radius_level=2)
    rbf.remapping(ll_f, cs_f)
    '''
    dst_size = len( ncf.dimensions['dst_size'] )
    src_address = ncf.variables['src_address'][:]
    remap_matrix = ncf.variables['remap_matrix'][:]

    for dst in xrange(dst_size):
        srcs = src_address[dst,:]
        invmat = remap_matrix[dst,:,:]

        wfs = np.dot(invmat, src_f[srcs])
        dst_f[dst] = fsum(wfs)


#----------------------------------------------------------
# Standard errors
#----------------------------------------------------------
ref_dst_f = np.zeros_like(dst_f)
for i, (lat,lon) in enumerate(dst_obj.latlons):
    ref_dst_f[i] = testfunc(lat, lon)

L1, L2, Linf = sem_1_2_inf(ref_dst_f, dst_f)
print ''
print 'L1= %e'%L1
print 'L2= %e'%L2
print 'Linf= %e'%Linf


#----------------------------------------------------------
# Plot with vtk
#----------------------------------------------------------
print ''
print 'Generate VTK file'

vtk_dir = '/nas/scteam/VisIt_data/remap/'

cs_vtk = CubeVTK2D(ne, ngq, rotated)
ll_vtk = LatlonVTK2D(nlat, nlon, ll_type, 'sphere')

if direction == 'll2cs':
    vll = (('ll_f', 1, 1, src_f.tolist()),)
    ll_vtk.write_with_variables(vtk_dir+'sph%d%d_ll_%dx%d_%s.vtk'%(m,n,nlat,nlon,ll_type), vll)

    vcs = (('ref_cs_f', 1, 1, ref_dst_f.tolist()), ('cs_f', 1, 1, dst_f.tolist()))
    fpath = vtk_dir + '%s/sph%d%d_%s_ne%d_%s_%dx%d_%s_%s.vtk'%(method, m, n, direction, ne, cs_type, nlat, nlon, ll_type, method)
    cs_vtk.write_with_variables(fpath, vcs)

else:
    vcs = (('cs_f', 1, 1, src_f.tolist()),)
    cs_vtk.write_with_variables(vtk_dir+'sph%d%d_cs_ne%d_%s.vtk'%(m,n,ne,cs_type), vcs)

    vll = (('ref_ll_f', 1, 1, ref_dst_f.tolist()), ('ll_f', 1, 1, dst_f.tolist()))
    fpath = vtk_dir + '%s/sph%d%d_%s_ne%d_%s_%dx%d_%s_%s.vtk'%(method, m, n, direction, ne, cs_type, nlat, nlon, ll_type, method)
    ll_vtk.write_with_variables(fpath, vll)
