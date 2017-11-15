'''

abstarct : Use the hand-written C library using numpy

history :
  2017-03-13  Ki-Hwan Kim  Start (ref. www.scipy-lectures.org)

'''


from __future__ import print_function, division
from ctypes import POINTER, c_int, byref

import numpy as np
import numpy.ctypeslib as npct
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal



class MyMath(object):
    '''
    Wrapper the math C library
    '''

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

        self.nx_p = byref(c_int(nx))
        self.ny_p = byref(c_int(ny))

        # Load the library using numpy
        libm = npct.load_library('my_math', './')
        my_cos = getattr(libm, 'my_cos_')
        my_cos_2d = getattr(libm, 'my_cos_2d_')

        # Set the argument and return type
        c_int_p = POINTER(c_int)

        arr_1d_f8 = npct.ndpointer(shape=(nx,), dtype='f8')
        my_cos.argtypes = (c_int_p, arr_1d_f8, arr_1d_f8)
        my_cos.restype = None

        arr_2d_f8 = npct.ndpointer(shape=(nx, ny), dtype='f8', flags='F')
        my_cos_2d.argtypes = (c_int_p, c_int_p, arr_2d_f8, arr_2d_f8)
        my_cos_2d.restype = None

        # Set to public
        self.libm = libm
        self.my_cos = my_cos
        self.my_cos_2d = my_cos_2d


    def cos(self, in_arr, out_arr):
        nx_p = self.nx_p

        return self.my_cos(nx_p, in_arr, out_arr)


    def cos2d(self, in_arr, out_arr):
        nx_p = self.nx_p
        ny_p = self.ny_p

        return self.my_cos_2d(nx_p, ny_p, in_arr, out_arr)



def main():
    '''
    main()
    '''

    nx, ny = 1000, 1200
    in_1d = np.random.rand(nx)
    out_1d = np.zeros(nx, 'f8')

    in_2d = np.random.rand(nx*ny).reshape((nx, ny), order='F')
    out_2d = np.zeros((nx, ny), 'f8', order='F')

    mymath = MyMath(nx, ny)
    mymath.cos(in_1d, out_1d)
    a_equal(out_1d, np.cos(in_1d))

    mymath.cos2d(in_2d, out_2d)
    a_equal(out_2d, np.cos(in_2d))



if __name__ == '__main__':
    main()
