'''

abstarct : Use the hand-written C library using numpy

history :
  2017-03-13  Ki-Hwan Kim  Start (ref. www.scipy-lectures.org)

'''


from __future__ import print_function, division
from ctypes import POINTER, c_int, c_double, byref

import numpy as np
import numpy.ctypeslib as npct
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal



class MyMath(object):
    '''
    Wrapper the math C library
    '''

    def __init__(self):
        # Load the library using numpy
        libm = npct.load_library('my_math', './')
        my_cos = getattr(libm, 'my_cos_')

        # Set the argument and return type
        my_cos.argtypes = [POINTER(c_double)]
        my_cos.restype = c_double

        # Set to public
        self.libm = libm
        self.my_cos = my_cos


    def cos(self, x):
        return self.my_cos(byref(c_double(x)))



def main():
    '''
    main()
    '''

    x = np.random.rand()

    mymath = MyMath()
    f = mymath.cos(x)
    a_equal(f, np.cos(x))



if __name__ == '__main__':
    main()
