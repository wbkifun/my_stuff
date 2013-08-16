from __future__ import division
import numpy
import h5py as h5
from pprint import pprint
from numpy.testing import assert_array_equal as assert_ae



#------------------------------------------------------------------------------
# variables
#------------------------------------------------------------------------------
nx, ny = 3, 4

arr_a = numpy.random.rand(nx,ny)
arr_b = numpy.random.rand(nx,ny)

print 'write arrays'
pprint(arr_a)
pprint(arr_b)


#------------------------------------------------------------------------------
# write a h5 file
#------------------------------------------------------------------------------
h5f = h5.File('write_read.h5', 'w')

# attributes
h5f.attrs['description'] = 'netCDF4 write/read test'     # string
h5f.attrs['size'] = nx*ny                                # number

# dataset
h5f.create_dataset('arr_a', data=arr_a, compression='gzip')
h5f.create_dataset('arr_b', data=arr_b, compression='gzip')

h5f.close()



#------------------------------------------------------------------------------
# read a nc file
#------------------------------------------------------------------------------
h5f = h5.File('write_read.h5', 'r')

print '='*80
print 'description', h5f.attrs['description']
print 'size', h5f.attrs['size']

read_a = h5f['arr_a'].value
read_b = h5f['arr_b'].value

print 'read arrays'
pprint(read_a)
pprint(read_b)


# verification
assert_ae(read_a, arr_a)
assert_ae(read_b, arr_b)

h5f.close()
