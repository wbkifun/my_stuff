#------------------------------------------------------------------------------
# filename  : compare_c_fortran_order.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.11.18    start
#
#
# description: 
#   Compare array sequences between C and Fortran ordering
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal




nx, ny, nz = 3, 4, 5

ac = np.zeros((nx,ny,nz), order='C')
af = np.zeros((nz,ny,nx), order='F')

for i in xrange(nx):
    for j in xrange(ny):
        for k in xrange(nz):
            ac[i,j,k] = i + 10*j + 100*k

for k in xrange(nz):
    for j in xrange(ny):
        for i in xrange(nx):
            af[k,j,i] = i + 10*j + 100*k

a_equal(ac.reshape(nx*ny*nz), af.reshape(nx*ny*nz, order='A'))
a_equal(ac.reshape(-1), af.reshape(-1, order='A'))  # view, it may be preferable in many cases
a_equal(ac.ravel(), af.T.ravel())
a_equal(ac.ravel(), af.ravel(order='A'))        # copy if needed
a_equal(ac.flatten(), af.flatten(order='A'))    # copy
