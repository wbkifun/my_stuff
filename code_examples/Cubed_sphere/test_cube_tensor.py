#------------------------------------------------------------------------------
# filename  : test_cube_tensor.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.25     start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
from math import fsum

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

from pkg.util.compare_float import feq




def test_jacobian_area_30_4():
    '''
    cs_tensor_ne30ngq4.nc: Jacobian area test
    '''
    ne, ngq = 30, 4

    cs_ncf = nc.Dataset('cs_grid_ne%dngq%d.nc'%(ne,ngq), 'r', format='NETCDF4')
    size = len( cs_ncf.dimensions['size'] )
    uids = cs_ncf.variables['uids'][:]
    mvps = cs_ncf.variables['mvps'][:]
    gq_indices = cs_ncf.variables['gq_indices'][:]

    ncf = nc.Dataset('cs_tensor_ne%dngq%d.nc'%(ne,ngq), 'r', format='NETCDF4')
    J = ncf.variables['J'][:]
    gq_wts = ncf.variables['gq_wts'][:]

    areas = np.zeros(size)
    for seq in xrange(size):
        panel, ei, ej, gi, gj = gq_indices[seq]
        u_seq = uids[seq]
        areas[seq] = J[u_seq]*gq_wts[gi-1]*gq_wts[gj-1]

    #aa_equal(fsum(areas), 4*np.pi, 11)
    #aa_equal(fsum(areas)/100, 4*np.pi/100, 13)  # /100 for accuracy comparison
    ok_( feq(fsum(areas), 4*np.pi, 13) )
