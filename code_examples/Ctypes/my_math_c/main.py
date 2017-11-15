'''

abstarct : Use the hand-written C library using numpy

history :
  2017-03-13  Ki-Hwan Kim  start (ref. www.scipy-lectures.org)
                           failed 2d array with **
  2017-09-04  Ki-Hwan Kim  add get_ptr(), 2d array with ** OK!

'''


from __future__ import print_function, division
from ctypes import c_int

import numpy as np
import numpy.ctypeslib as npct
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal



def get_ptr(x):
    xpp = (x.__array_interface__['data'][0] 
           + np.arange(x.shape[0])*x.strides[0]).astype(np.uintp)

    return xpp



class MyMath(object):
    '''
    Wrapper the math C library
    '''

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

        # Load the library using numpy
        libm = npct.load_library('my_math', './')

        # Set the argument and return type
        f1d = npct.ndpointer(ndim=1, dtype='f8')
        libm.my_cos.argtypes = [c_int, f1d, f1d]
        libm.my_cos.restype = None

        libm.my_cos_2d.argtypes = [c_int, c_int, f1d, f1d]
        libm.my_cos_2d.restype = None

        pp = npct.ndpointer(ndim=1, dtype=np.uintp, flags='C')
        libm.my_cos_2d_2ptr.argtypes = [c_int, c_int, pp, pp]
        libm.my_cos_2d_2ptr.restype = None

        # Set public
        self.libm = libm


    def cos(self, in_arr, out_arr):
        nx = self.nx

        return self.libm.my_cos(nx, in_arr, out_arr)


    def cos2d(self, in_arr, out_arr):
        nx = self.nx
        ny = self.ny

        return self.libm.my_cos_2d(nx, ny, in_arr.ravel(), out_arr.ravel())


    def cos2d_2ptr(self, in_arr, out_arr):
        nx = self.nx
        ny = self.ny

        return self.libm.my_cos_2d_2ptr(nx, ny, get_ptr(in_arr), get_ptr(out_arr))



def main():
    '''
    main()
    '''

    nx, ny = 1000, 1000
    in_1d = np.random.rand(nx)
    out_1d = np.zeros(nx, 'f8')

    in_2d = np.random.rand(nx, ny)
    out_2d = np.zeros((nx, ny), 'f8', order='C')
    #out_2d = np.zeros(nx*ny, 'f8').reshape(nx,ny)

    mymath = MyMath(nx, ny)
    mymath.cos(in_1d, out_1d)
    a_equal(out_1d, np.cos(in_1d))

    mymath.cos2d(in_2d, out_2d)
    a_equal(out_2d, np.cos(in_2d))

    mymath.cos2d_2ptr(in_2d, out_2d)
    a_equal(out_2d, np.cos(in_2d))



if __name__ == '__main__':
    main()
