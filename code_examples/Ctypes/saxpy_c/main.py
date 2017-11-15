'''

abstarct : Use the hand-written C library using numpy

history :
  2017-03-13  Ki-Hwan Kim  start (ref. www.scipy-lectures.org)
                           failed 2d array with **
  2017-09-04  Ki-Hwan Kim  add get_ptr(), 2d array with ** OK!

'''


from __future__ import print_function, division
from ctypes import c_int, c_float
from datetime import datetime

import numpy as np
import numpy.ctypeslib as npct
from numpy.testing import assert_array_equal as a_equal



def saxpy_numpy(a, x, y):
    y[:] = a*x + y


class SAXPY:
    def __init__(self):
        # load the library using numpy
        libm = npct.load_library('saxpy', './')
        cfunc = getattr(libm, 'saxpy')

        # set the arguments and retun types
        arr_f4 = npct.ndpointer(ndim=1, dtype='f4')
        cfunc.argtypes = [c_int, c_float, arr_f4, arr_f4]
        cfunc.restype = None

        # set public
        self.cfunc = cfunc

    def saxpy_c(self, n, a, x, y):
        self.cfunc(n, a, x, y)


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
    obj.saxpy_c(n, a, x, y2)
    dt_c = datetime.now() - t2

    print('n={}'.format(n))
    print('numpy: {}'.format(dt_numpy))
    print('c    : {}'.format(dt_c))

    a_equal(y, y2)
    print('Check result: OK!)


if __name__ == '__main__':
    main()
