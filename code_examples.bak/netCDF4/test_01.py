#!/usr/bin/env python

from __future__ import division
import logging
import numpy
import netCDF4 as nc


#------------------------------------------------------------------------------
# print debug messages using logging
#------------------------------------------------------------------------------
#logging.basicConfig(level=logging.DEBUG, \
#        format='%(pathname)s\n%(asctime)s %(levelname)s: line %(lineno)s: %(message)s', \
#        datefmt='%Y-%m-%d %H:%M:%S')


#------------------------------------------------------------------------------
# setup
#------------------------------------------------------------------------------
np, ne = 4, 10
v0_ref = numpy.zeros(np, int)
v1_ref = numpy.zeros((np,np), int, order='F')
v2_ref = numpy.zeros((np,np,ne), int, order='F')

for i in xrange(np):
    v0_ref[i] = i+1

for j in xrange(np):
    for i in xrange(np):
        v1_ref[i,j] = (i+1) + 100*(j+1)

for ie in xrange(ne):
    for j in xrange(np):
        for i in xrange(np):
            v2_ref[i,j,ie] = (i+1) + 100*(j+1) + 10000*(ie+1)


#------------------------------------------------------------------------------
# read a nc file
#------------------------------------------------------------------------------
f = nc.Dataset('nc_test_01.nc', 'r')

v0 = f.variables['v0'][:]
v1 = f.variables['v1'][:].reshape(np*np).reshape((np,np), order='F')
v2 = f.variables['v2'][:].reshape(np*np*ne).reshape((np,np,ne), order='F')
f.close()

assert numpy.linalg.norm(v0_ref - v0) == 0
assert numpy.linalg.norm(v1_ref - v1) == 0
assert numpy.linalg.norm(v2_ref - v2) == 0

