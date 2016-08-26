#------------------------------------------------------------------------------
# filename  : compare_float_array.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.4.5     start
#
# description
#   compare two float array using numpy.frexp
#------------------------------------------------------------------------------

import numpy as np
import re
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal




def compare_float(var, ref, cut_digit=1, verbose=False):
    # x = mantissa * 2**exponent

    if type(var) != np.ndarray: var = np.array(var)
    if type(ref) != np.ndarray: ref = np.array(ref)

    m_var, e_var = np.frexp(var)
    m_ref, e_ref = np.frexp(ref)

    num_exp_diff = np.count_nonzero(e_var != e_ref)
    if num_exp_diff > 0:
        try:
            # check 0 and 1e-16
            idxs = np.where(e_var != e_ref)
            aa_equal(var[idxs], ref[idxs], 15)
            return True, 15
        except:
            if verbose:
                print("idxs   : ", idxs)
                print("Actual : ", var[idxs])
                print("Desired: ", ref[idxs])
            return False, "The exponents are not same at {} points".format(num_exp_diff)

    num_man_diff = np.count_nonzero(m_var != m_ref)
    if num_man_diff == 0:
        return True, "exact"

    digit = 17
    percents = []
    while(True):
        try:
            aa_equal(m_var, m_ref, digit-1)
            return True, "{}, ({})".format(digit, ', '.join(percents))

        except Exception as e:
            percent = float(re.findall('mismatch (\d+.\S+)%',str(e))[0])
            percents.insert(0, "{}:{:.2f}%".format(digit,percent))
            #print('>>>>', digit, str(e), percent, percents)

            if digit == cut_digit:
                return False, "{}, ({})".format(cut_digit, ', '.join(percents))
            else:
                digit -= 1
