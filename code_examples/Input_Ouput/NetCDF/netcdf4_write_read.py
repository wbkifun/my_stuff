from __future__ import print_function, division
from numpy.random import rand
from netCDF4 import Dataset
from pprint import pprint
from numpy.testing import assert_array_equal as assert_ae



#------------------------------------------------------------------------------
# variables
#------------------------------------------------------------------------------
nx, ny = 3, 4

arr_a = rand(nx,ny)
arr_b = rand(nx,ny)

print('write arrays')
pprint(arr_a)
pprint(arr_b)


#------------------------------------------------------------------------------
# write a nc file
#------------------------------------------------------------------------------
ncf = Dataset('write_read.nc', 'w', format='NETCDF4')

# attributes
ncf.description = 'netCDF4 write/read test'     # string
ncf.size = nx*ny                                # number

# dimensions
ncf.createDimension('nx', nx)
ncf.createDimension('ny', ny)

# variables
va = ncf.createVariable('va', 'f8', ('nx','ny'))
va.unit = 'm/s'
vb = ncf.createVariable('vb', 'f8', ('nx','ny'))
vb.unit = 'kg'

# write data to variables
va[:] = arr_a
vb[:] = arr_b

ncf.close()



#------------------------------------------------------------------------------
# read a nc file
#------------------------------------------------------------------------------
ncf = Dataset('write_read.nc', 'r', format='NETCDF4')

print('='*80)
print('description', ncf.description)
print('size', ncf.size)

dim_nx = len( ncf.dimensions['nx'] )
dim_ny = len( ncf.dimensions['ny'] )

read_a = ncf.variables['va'][:]
read_b = ncf.variables['vb'][:]

print('read arrays')
pprint(read_a)
pprint(read_b)


# verification
assert_ae(read_a, arr_a)
assert_ae(read_b, arr_b)

ncf.close()
