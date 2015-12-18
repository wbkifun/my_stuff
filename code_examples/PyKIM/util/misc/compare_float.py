#------------------------------------------------------------------------------
# filename  : compare_float.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.8.21    (add is_between() and is_between_angle())
#             2013.6.26    (rename)
#             2015.2.5     add feq_fixed()
#
# subroutines to compare the floating-point numbers
#   feq_fixed()
#   feq()
#   fne()
#   flt()
#   fgt()
#   fle()
#   fge()
#   flgt()
#   flge()
#   is_between()
#   is_between_angle()
#------------------------------------------------------------------------------

from __future__ import division
import numpy
from numpy import fabs, pi
from numpy.testing import assert_approx_equal as assert_ape
from numpy.testing import assert_almost_equal as assert_ale


DIGIT = 15




def feq_fixed(a, b, digit=DIGIT):
    # ? a == b with leading zero
    try:
        assert_ale(a, b, digit)
    except AssertionError:
        return False
    else:
        return True




def feq(a, b, digit=DIGIT):
    # ? a == b without leading zeros

    if fabs(b) < pow(10,-digit+1):      # b == 0
        if fabs(a) < pow(10,-digit+1):  # a == 0
            return True
        else:
            return False

    elif fabs(a) < pow(10,-digit+1):    # a == 0
        if fabs(b) < pow(10,-digit+1):  # b == 0
            return True
        else:
            return False

    else:
        try:
            assert_ape(a, b, digit)
        except AssertionError:
            return False
        else:
            return True




def fne(a, b, digit=DIGIT):
    # ? a != b

    return not feq(a, b, digit)




def flt(a, b, digit=DIGIT):
    # ? a < b

    if feq(a, b, digit):
        return False
    elif a > b:
        return False
    else:
        return True




def fgt(a, b, digit=DIGIT):
    # ? a > b

    if feq(a, b, digit):
        return False
    elif a < b:
        return False
    else:
        return True




def fle(a, b, digit=DIGIT):
    # ? a <= b

    if feq(a, b, digit):
        return True
    elif a < b:
        return True
    else:
        return False




def fge(a, b, digit=DIGIT):
    # ? a >= b

    if feq(a, b, digit):
        return True
    elif a > b:
        return True
    else:
        return False




def flgt(a, b, c, digit=DIGIT):
    # ? a < b < c

    if flt(a, b, digit) and flt(b, c, digit):
        return True
    else:
        return False




def flge(a, b, c, digit=DIGIT):
    # ? a <= b <= c

    if fle(a, b, digit) and fle(b, c, digit):
        return True
    else:
        return False




def is_between(a, b, c, include=True, digit=DIGIT):
    # ? min(a,c) <= b <= max(a,c)

    flx = fle if include else flt

    if flx(min(a,c), b, digit) and flx(b, max(a,c), digit):
        return True
    else:
        return False




def is_between_angle(angle1, angle, angle2, include=True, digit=DIGIT):
    '''
    ? angle1 <= angle <= angle2  or  angle2 <= angle <= angle1
    '''

    a12 = fabs(angle1 - angle2)
    a1 = fabs(angle - angle1)
    a2 = fabs(angle - angle2)

    # when lon1 and lon2 are right-and-left lon=0
    if fgt(a12, pi): a12 = 2*pi - a12
    if fgt(a1, pi): a1 = 2*pi - a1
    if fgt(a2, pi): a2 = 2*pi - a2


    if include:
        if feq(a1,0,digit) or feq(a2,0,digit) or \
           feq(a12, (a1+a2), digit):
            return True
        else:
            return False

    else:
        if fne(a1,0,digit) and fne(a2,0,digit) and \
           feq(a12, (a1+a2), digit):
            return True
        else:
            return False
