from __future__ import division
import numpy
from numpy.testing import assert_array_equal as assert_ae
import netCDF4 as nc



#------------------------------------------------------------------------------
# variables
#------------------------------------------------------------------------------
nx, ny = 5, 7

data = numpy.arange(nx*ny, dtype='f8')
data_c = data.reshape((5,7), order='C')
data_f = data.reshape((5,7), order='F')



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

print f.description

dim_nx = len( f.dimensions['nx'] )
dim_ny = len( f.dimensions['ny'] )

read_c = f.variables['vc'][:]
read_f = f.variables['vf'][:]
read_f2 = f.variables['vf'][:].reshape((dim_nx, dim_ny), order='F')

print data_c.flatten()
print data_f.flatten()
print '-'*47
print read_f.flatten()
print '-'*47
print read_c.flatten()
print read_f2.flatten()

assert_ae(data_c, read_c)
assert_ae(data_f, read_f)
assert_ae(data_f, read_f2)

f.close()
