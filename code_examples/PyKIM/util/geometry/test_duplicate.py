import numpy as np
from numpy import pi, sqrt, sin
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.extend([current_dpath,dirname(current_dpath)])




def test_duplicate_idxs():
    '''
    duplicate_idxs() : normal case, latlons
    '''
    from duplicate import duplicate_idxs


    # normal case
    xyzs = [(0,0.5,0),(-0.5,0,0),(0.5,0,0),(1,1.2,0),(0.5,0,0)]
    ret = duplicate_idxs(xyzs)
    equal(ret, [4])

    xyzs = [(0,0.5,0),(-0.5,0,0),(0.5,0,0),(1,1.2,0),(0.5,0,0),(1.2,2.3,0),(-0.5,0,0)]
    ret = duplicate_idxs(xyzs)
    equal(ret, [6,4])


    # error case
    xyzs = [(-0.69766285707571141, 1.5271630954950388, 0), \
            (-0.69766285707571152, 1.5271630954950384, 0), \
            (-0.78492204764598783, 1.5271630954950381, 0), \
            (-0.78492204764598794, 1.5271630954950384, 0), \
            (-0.78492204764598794, 1.6144295580947545, 0), \
            (-0.69766285707571218, 1.6144295580947559, 0), \
            (-0.69766285707571163, 1.6144295580947547, 0)]
    ret = duplicate_idxs(xyzs)
    equal(ret, [1,3,6])




def test_remove_duplicates():
    '''
    remove_duplicates() : normal case, latlons
    '''
    from duplicate import remove_duplicates


    # normal case
    xyzs = [(0,0.5,0),(-0.5,0,0),(0.5,0,0),(1,1.2,0),(0.5,0,0)]
    ret = remove_duplicates(xyzs)
    equal(ret, [(0,0.5,0),(-0.5,0,0),(0.5,0,0),(1,1.2,0)])

    xyzs = [(0,0.5,0),(-0.5,0,0),(0.5,0,0),(1,1.2,0),(0.5,0,0),(1.2,2.3,0),(-0.5,0,0)]
    ret = remove_duplicates(xyzs)
    equal(ret, [(0,0.5,0),(-0.5,0,0),(0.5,0,0),(1,1.2,0),(1.2,2.3,0)])
