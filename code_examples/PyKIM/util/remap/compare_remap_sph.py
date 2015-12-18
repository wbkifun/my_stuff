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
method = 'vgecore'      # 'bilinear', 'vgecore', 'lagrange'
direction = 'll2cs'
cs_type = 'regular'     # 'regular', 'rotated'
ll_type = 'regular'     # 'regular', 'gaussian'

#ne, ngq = 15, 4
ne, ngq = 30, 4
#ne, ngq = 60, 4
#ne, ngq = 120, 4
rotated = cs_type == 'rotated'
cs_obj = CubeGridRemap(ne, ngq, rotated)

#nlat, nlon = 90, 180
nlat, nlon = 180, 360
#nlat, nlon = 360, 720
#nlat, nlon = 720, 1440

#nlat, nlon = 192, 384
ll_obj = LatlonGridRemap(nlat, nlon, ll_type)


m, n = 16, 32
ll_f = np.zeros(ll_obj.nsize, 'f8')
for i, (lat,lon) in enumerate(ll_obj.latlons):
    ll_f[i] = sph_harm(m, n, lon, np.pi/2-lat).real


print ''
print 'ne=%d, ngq=%d, %s'%(ne, ngq, cs_type)
print 'nlat=%d, nlon=%d, %s'%(nlat, nlon, ll_type)
print 'method: %s'%(method)
print 'direction: %s'%(direction)
print 'SPH m=%d, n=%d'%(m, n)


#----------------------------------------------------------
# Remap
#----------------------------------------------------------
cs_f = np.zeros(cs_obj.up_size, ll_f.dtype)

remap_dir = '/nas2/user/khkim/remap_matrix/'
fname = 'remap_%s_ne%d_%s_%dx%d_%s_%s.nc'%(direction, ne, cs_type, nlat, nlon, ll_type, method)

ncf = nc.Dataset(remap_dir+fname, 'r', 'NETCDF3_CLASSIC')
num_links = len( ncf.dimensions['num_links'] )
dsts = ncf.variables['dst_address'][:]
srcs = ncf.variables['src_address'][:]
wgts = ncf.variables['remap_matrix'][:]

for dst, src, wgt in zip(dsts, srcs, wgts):
    cs_f[dst] += ll_f[src]*wgt


#----------------------------------------------------------
# Standard errors
#----------------------------------------------------------
ref_cs_f = np.zeros_like(cs_f)
for i, (lat,lon) in enumerate(cs_obj.latlons):
    ref_cs_f[i] = sph_harm(m, n, lon, np.pi/2-lat).real

L1, L2, Linf = sem_1_2_inf(ref_cs_f, cs_f)
print ''
print 'L1', L1
print 'L2', L2
print 'Linf', Linf


#----------------------------------------------------------
# Plot with vtk
#----------------------------------------------------------
vtk_dir = '/nas/scteam/VisIt_data/remap/'
fpath = vtk_dir + '%s/sph%d%d_%s_ne%d_%s_%dx%d_%s_%s.nc'%(method, m, n, direction, ne, cs_type, nlat, nlon, ll_type, method)

ll_vtk = LatlonVTK2D(nlat, nlon, ll_type, 'sphere')
vll = (('ll_f', 1, 1, ll_f.tolist()),)
ll_vtk.write_with_variables(vtk_dir+'sph%d%d_ll_%dx%d_%s.vtk'%(m,n,nlat,nlon,ll_type), vll)

cs_vtk = CubeVTK2D(ne, ngq, rotated)
vcs = (('ref_cs_f', 1, 1, ref_cs_f.tolist()), ('cs_f', 1, 1, cs_f.tolist()))
cs_vtk.write_with_variables(fpath, vcs)
