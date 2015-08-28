from __future__ import division
import numpy
from numpy.testing import assert_array_equal as assert_ae
import netCDF4 as nc



#------------------------------------------------------------------------------
# variables
#------------------------------------------------------------------------------
nx, ny = 3, 4

data = numpy.arange(nx*ny, dtype='f8')
data_c = data.reshape((nx,ny), order='C')
data_f = data.reshape((nx,ny), order='F')

print "data_c.ravel('A')", data_c.ravel('A')
print "data_c.ravel('C')", data_c.ravel('C')
print "data_c.ravel('F')", data_c.ravel('F')
print '-'*80
print "data_f.ravel('A')", data_f.ravel('A')
print "data_f.ravel('C')", data_f.ravel('C')
print "data_f.ravel('F')", data_f.ravel('F')


#------------------------------------------------------------------------------
# write a nc file
#------------------------------------------------------------------------------
grp = nc.Dataset('write_read.nc', 'w', format='NETCDF4')

# attributes
grp.description = 'netCDF4 write/read test'

# dimensions
nx_d = grp.createDimension('nx', nx)
ny_d = grp.createDimension('ny', ny)

# variables
vc_v = grp.createVariable('vc', 'f8', ('nx','ny'))
vf_v = grp.createVariable('vf', 'f8', ('nx','ny'))

# write data to variables
vc_v[:] = data_c
vf_v[:] = data_f

grp.close()



#------------------------------------------------------------------------------
# read a nc file
#------------------------------------------------------------------------------
f = nc.Dataset('write_read.nc', 'r', format='NETCDF4')

print '='*80
print f.description

dim_nx = len( f.dimensions['nx'] )
dim_ny = len( f.dimensions['ny'] )

read_c = f.variables['vc'][:]
read_f = f.variables['vf'][:]
print 'read_c.shape', read_c.shape
print 'read_f.shape', read_f.shape

read_f2 = f.variables['vf'][:].reshape((dim_nx, dim_ny), order='F')
read_f3 = numpy.zeros((dim_nx, dim_ny), order='F')
read_f3[:] = f.variables['vf'][:]

print "read_c.ravel('A')", read_c.ravel('A')
print "read_c.ravel('C')", read_c.ravel('C')
print "read_c.ravel('F')", read_c.ravel('F')
print '-'*80
print "read_f.ravel('A')", read_f.ravel('A')
print "read_f.ravel('C')", read_f.ravel('C')
print "read_f.ravel('F')", read_f.ravel('F')
print '-'*80
print "read_f2.ravel('A')", read_f2.ravel('A')
print "read_f2.ravel('C')", read_f2.ravel('C')
print "read_f2.ravel('F')", read_f2.ravel('F')
print '-'*80
print "read_f3.ravel('A')", read_f3.ravel('A')
print "read_f3.ravel('C')", read_f3.ravel('C')
print "read_f3.ravel('F')", read_f3.ravel('F')
print '-'*80

assert_ae(data_c.ravel('A'), read_c.ravel('A'))
assert_ae(data_f.ravel('A'), read_f3.ravel('A'))

f.close()
