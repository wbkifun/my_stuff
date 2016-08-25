#------------------------------------------------------------------------------
# filename  : test_compare_float_array.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.4.5  start
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_compare_float():
    from compare_float_array import compare_float

    a = np.float64(['0.12345678901234567','12345.678901234567','0.00012345678901234567'])
    b = np.float64(['0.12345678901234567','12345.678901234567','0.00012345678901234567'])
    tf, msg = compare_float(a, b)
    equal((tf, msg), (True, "exact"))


    a = np.float64(['0.12345678901234567','12345.678901234567','0.00012345678901234567'])
    b = np.float64(['1.2345678901234567' ,'123.45678901234567','0.00012345678901234567'])
    tf, msg = compare_float(a, b)
    equal((tf, msg), (False, "The exponents are not same at 2 points"))


    a = np.float64(['0.12345678901234567','12345.678901234567','0.00012345678901234567'])
    b = np.float64(['0.12345678901234598','12345.678901234567','0.00012345678901234567'])
    tf, msg = compare_float(a, b)
    equal((tf, msg), (True, "15, (16:33.33%, 17:33.33%)"))


    a = np.float64(['0.12345678901234567','12345.678901234567','0.00012345678901234567'])
    b = np.float64(['0.12345678901234598','12345.678901234987','0.00012345678901234567'])
    tf, msg = compare_float(a, b)
    equal((tf, msg), (True, "14, (15:33.33%, 16:66.67%, 17:66.67%)"))


    a = np.float64(['0.12345678901234567','12345.678901234567','0.00012345678901234567'])
    b = np.float64(['0.12345678901234598','12345.678901234987','0.00012345678901239876'])
    tf, msg = compare_float(a, b)
    equal((tf, msg), (True, "13, (14:33.33%, 15:66.67%, 16:100.00%, 17:100.00%)"))


    a = np.float64(['0.12345678901234567','12345.678901234567','0.00012345678901234567'])
    b = np.float64(['0.12345678901234598','12345.678901234987','0.00012345678901239876'])
    tf, msg = compare_float(a, b)
    equal((tf, msg), (True, "13, (14:33.33%, 15:66.67%, 16:100.00%, 17:100.00%)"))


    a = np.float64(['0.12345678901234567'])
    b = np.float64(['0.12345678901234987'])
    tf, msg = compare_float(a, b)
    equal((tf, msg), (True, "14, (15:100.00%, 16:100.00%, 17:100.00%)"))
