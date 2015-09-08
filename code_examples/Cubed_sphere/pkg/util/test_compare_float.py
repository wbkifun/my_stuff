#------------------------------------------------------------------------------
# filename  : test_compare_float.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.4   revise
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_feq_fixed_float64():
    from compare_float import feq_fixed

    a = np.float64('0.123456789012345678')
    b = np.float64('0.123456789012345901')
    assert feq_fixed(a, b, 15) == True
    assert feq_fixed(a, b, 16) == False


    a = np.float64('0.00123456789012345678')
    b = np.float64('0.00123456789012301234')
    assert feq_fixed(a, b, 15) == True
    assert feq_fixed(a, b, 16) == False


    a = np.float64('0.00000123456789012345678')
    b = np.float64('0.00000123456789001234567')
    assert feq_fixed(a, b, 15) == True
    assert feq_fixed(a, b, 16) == False


    a = np.float64('1.23456789012345678')
    b = np.float64('1.23456789012345901')
    assert feq_fixed(a, b, 14) == True
    assert feq_fixed(a, b, 15) == False


    a = np.float64('123456.789012345678')
    b = np.float64('123456.789012345901')
    assert feq_fixed(a, b, 9) == True
    assert feq_fixed(a, b, 10) == False




def test_feq_float64():
    from compare_float import feq

    a = np.float64('0.123456789012345678')
    b = np.float64('0.123456789012345901')
    assert feq(a, b, 15) == True
    assert feq(a, b, 16) == False


    a = np.float64('0.00123456789012345678')
    b = np.float64('0.00123456789012345901')
    assert feq(a, b, 15) == True
    assert feq(a, b, 16) == False


    a = np.float64('0.00000123456789012345678')
    b = np.float64('0.00000123456789012345901')
    assert feq(a, b, 15) == True
    assert feq(a, b, 16) == False


    a = np.float64('1.23456789012345678')
    b = np.float64('1.23456789012345901')
    assert feq(a, b, 15) == True
    assert feq(a, b, 16) == False


    a = np.float64('123456.789012345678')
    b = np.float64('123456.789012345901')
    assert feq(a, b, 15) == True
    assert feq(a, b, 16) == False




def test_feq_float32():
    from compare_float import feq

    a = np.float32('0.123456789012345678')
    b = np.float32('0.123456789012345901')
    assert feq(a, b, 15) == True
    assert feq(a, b, 16) == True


    a = np.float32('0.123456789012')
    b = np.float32('0.123456789703')
    assert feq(a, b, 9) == True
    assert feq(a, b, 10) == True


    a = np.float32('0.12345678901')
    b = np.float32('0.12345678123')
    assert feq(a, b, 8) == True
    assert feq(a, b, 9) == False

    # => 32 bit float number has 9 digit.


    a = np.float32('12345.678901')
    b = np.float32('12345.678123')
    assert feq(a, b, 8) == True
    assert feq(a, b, 9) == False




def test_feq_zero():
    from compare_float import feq

    a = np.float64('0.123456789012345678')
    b = np.float64('0.000000000000000521')
    assert feq(b, 0, 15) == True
    assert feq(b, 0, 16) == True
    assert feq(b, 0, 17) == False


    a = np.float64('0.123456789012345678')
    b = np.float64('0.000006789012345678')
    assert feq(b, 0, 5) == True
    assert feq(b, 0, 6) == True
    assert feq(b, 0, 7) == False
    assert feq(b, 0, 8) == False


    a = np.float64('0.123456789012345678')
    b = np.float64('8')
    assert feq(b, 0, 5) == False


    assert feq(8, 0, 5) == False



def test_failed_case():
    from compare_float import feq, flge


    #---------------------------------------------
    beta0 = 0.7636902920551788
    beta  = 0.76369029205517769
    assert feq(beta0, beta, 15) == False
    assert feq(beta0, beta, 14) == True


    #---------------------------------------------
    a = 0.078539816339744856
    b = 0.07853981633974483
    assert feq(a, b, 15) == True


    #---------------------------------------------
    a0, b0 = -0.026179938779915091, -0.75938935817214992
    a1, a2 = -0.078539816339744856, -0.021707871342270618
    b1, b2 = -0.76369029205517769, -0.70685834705770345
    assert flge(a1, a0, a2) == True
    assert flge(b1, b0, b2) == True


    #---------------------------------------------
    a = -0.10700554748256769
    b = -0.10700554748256959885
    assert feq(a, b, 15) == False
    assert feq(a, b, 14) == True
