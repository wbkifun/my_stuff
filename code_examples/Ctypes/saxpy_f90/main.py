'''

abstarct : Use the hand-written Fortran library using numpy

history :
  2017-09-04  Ki-Hwan Kim  Start (ref. www.scipy-lectures.org)

'''


from __future__ import print_function, division
from ctypes import POINTER, c_int, c_float, byref
from datetime import datetime

import numpy as np
import numpy.ctypeslib as npct
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal



def saxpy_numpy(a, x, y):
    y[:] = a*x + y


class SAXPY(object):
    '''
    Wrapper the Fortran library
    '''

    def __init__(self):
        # Load the library using numpy
        libm = npct.load_library('saxpy', './')

        # Set the argument and return type
        c_int_p = POINTER(c_int)
        c_float_p = POINTER(c_float)

        arr_1d_f4 = npct.ndpointer(ndim=1, dtype='f4')
        libm.saxpy_.argtypes = (c_int_p, c_float_p, arr_1d_f4, arr_1d_f4)
        libm.saxpy_.restype = None

        # Set to public
        self.libm = libm


    def saxpy_f90(self, n, a, x, y):
        self.libm.saxpy_(byref(c_int(n)), byref(c_float(a)), x, y) 



def main():
    n = 2**25
    a = np.float32(np.random.rand())
    x = np.random.rand(n).astype('f4')
    y = np.random.rand(n).astype('f4')
    y2 = y.copy()

    t1 = datetime.now()
    saxpy_numpy(a, x, y)
    dt_numpy = datetime.now() - t1

    obj = SAXPY()
    t2 = datetime.now()
    obj.saxpy_f90(n, a, x, y2)
    dt_f90 = datetime.now() - t2

    print('n={}'.format(n))
    print('numpy: {}'.format(dt_numpy))
    print('f90  : {}'.format(dt_f90))

    a_equal(y, y2)
    print('Check result: OK!')


if __name__ == '__main__':
    main()
