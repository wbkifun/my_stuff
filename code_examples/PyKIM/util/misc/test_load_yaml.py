import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_replace_braket():
    from load_yaml import replace_braket

    conf_dict = { \
            'a': 4.5, \
            'b': 3.2, \
            'c': '3.e5', \
            'd': '<a>+<b>', \
            'e': '<a>/<d>'}

    for k, v in conf_dict.items():
        conf_dict[k] =  replace_braket(conf_dict, k)

    ref_dict = {'a':4.5, 'b':3.2, 'c':3e5, 'd':4.5+3.2, 'e':4.5/(4.5+3.2)}

    for k, v in conf_dict.items():
        equal(v, ref_dict[k])




def test_load_yaml_dict():
    from load_yaml import load_yaml_dict

    conf_dict = load_yaml_dict('sample.yaml')

    ref_dict = {'aa':5, 'bb':2, 'cc':3, \
            'section':{'dd':90., 'ee':5, 'ff':18.}, \
            'parameter':{'a':4.5, 'b':3.2, 'c':3e5, 'd':4.5+3.2, 'e':4.5/(4.5+3.2)}}

    for k, v in conf_dict.items():
        equal(v, ref_dict[k])
