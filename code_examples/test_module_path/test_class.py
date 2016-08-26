from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_



def test_class_a():
    from dir_a.class_a import CA

    ca = CA(1.2)
    equal(ca.a, 1.2)
    equal(ca.f, '/home/khkim/my_stuff/code_examples/test_module_path/dir_a/class_a.py')



def test_class_b():
    from dir_b.class_b import CB

    cb = CB(1.5)
    #equal(cb.a, 1.5)
    equal(cb.f, '/home/khkim/my_stuff/code_examples/test_module_path/dir_a/class_a.py')
