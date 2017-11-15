'''

abstarct : Use the math C library, m.so

history :
  2017-03-13  Ki-Hwan Kim  Start (ref. www.scipy-lectures.org)

'''


from __future__ import print_function, division
from ctypes.util import find_library
from random import uniform
from math import cos
import ctypes

from numpy.testing import assert_equal as equal



#class CMath(Object):
class CMath:
    '''
    Wrapper the math C library
    '''

    def __init__(self):
        # Find and load the library
        libm = ctypes.cdll.LoadLibrary(find_library('m'))

        # Set the argument and return type
        libm.cos.argtypes = [ctypes.c_double]
        libm.cos.restype = ctypes.c_double

        self.libm = libm


    def cos(self, arg):
        '''
        Wrapper for cos() from math.h
        '''

        return self.libm.cos(arg)



def main():
    '''
    main()
    '''

    cmath = CMath()

    x = uniform(-1, 1)
    equal(cmath.cos(x), cos(x))



if __name__ == '__main__':
    main()
