import numpy as np
import os
from numpy import pi, sqrt, sin, cos
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_


import io
import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.append(current_dpath)




def capture(func):
    def wrapper(*args, **kwargs):
        capturer1 = io.StringIO()
        capturer2 = io.StringIO()
        old_stdout, sys.stdout = sys.stdout, capturer1
        old_stderr, sys.stderr = sys.stderr, capturer2

        ret = func(*args, **kwargs)

        sys.stdout, sys.stderr = old_stdout, old_stderr
        out = capturer1.getvalue().rstrip('\n')
        err = capturer2.getvalue().rstrip('\n')

        return ret, out, err

    return wrapper




def test_compile_using_f2py_f90():
    '''
    compile_using_f2py: add.f90
    '''
    from source_module import compile_using_f2py, get_module_from_file

    # compile and import
    src = '''
SUBROUTINE add(nx, a, b, c)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx
  REAL(8), DIMENSION(nx), INTENT(IN) :: a, b
  REAL(8), DIMENSION(nx), INTENT(INOUT) :: c

  INTEGER :: ii

  DO ii=1,nx
    c(ii) = a(ii) + b(ii)
  END DO
END SUBROUTINE
    '''

    code_type = 'f90'
    dpath = os.path.join(current_dpath, 'src')
    build_dpath = os.path.join(dpath, 'build')
    src_fpath = os.path.join(dpath, 'add.'+code_type)
    with open(src_fpath, 'w') as f: f.write(src)
    ret, out, err = capture(compile_using_f2py)(src_fpath, compiler='gnu')
    mod = get_module_from_file(build_dpath, 'add', code_type)

    # setup
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    mod.add(nx, a, b, c)
    a_equal(a+b, c)




def test_compile_using_f2py_c():
    '''
    compile_using_f2py: add.c
    '''
    from source_module import compile_using_f2py, get_module_from_file

    # compile and import
    src = '''
void add(int nx, double *a, double *b, double *c) {
    // size and intent of array arguments for f2py
    // a :: nx, in
    // b :: nx, in
    // c :: nx, inout

    int i;

    for (i=0; i<nx; i++) {
        c[i] = a[i] + b[i];
    }
}
    '''

    code_type = 'c'
    dpath = os.path.join(current_dpath, 'src')
    build_dpath = os.path.join(dpath, 'build')
    src_fpath = os.path.join(dpath, 'add.'+code_type)
    with open(src_fpath, 'w') as f: f.write(src)
    ret, out, err = capture(compile_using_f2py)(src_fpath, compiler='gnu')
    mod = get_module_from_file(build_dpath, 'add', code_type)

    # setup
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    mod.add(nx, a, b, c)
    a_equal(a+b, c)




def test_make_signature_c():
    '''
    make_signature_c()
    '''
    from source_module import make_signature_c

    src_c = '''
void add(int nx, double *a, double *b, double *c) {
    // size and intent of array arguments for f2py
    // a :: nx, in
    // b :: nx, in
    // c :: nx, inout

    int i;

    for (i=0; i<nx; i++) {
        c[i] = a[i] + b[i];
    }
}

void dummy(int nx, double k, double *a, double *b) {
    // size and intent of array arguments for f2py
    // a :: nx, in
    // b :: nx, inout

    int i;

    for (i=0; i<nx; i++) {
        b[i] = a[i] + k*b[i];
    }
}
    '''

    ref_sig_c = '''
python module $MODNAME
  interface
    subroutine add(nx, a, b, c)
      intent(c) :: add
      intent(c)
      integer, required, intent(in) :: nx
      real(8), dimension(nx), intent(in) :: a
      real(8), dimension(nx), intent(in) :: b
      real(8), dimension(nx), intent(inout) :: c
    end subroutine
    subroutine dummy(nx, k, a, b)
      intent(c) :: dummy
      intent(c)
      integer, required, intent(in) :: nx
      real(8), intent(in) :: k
      real(8), dimension(nx), intent(in) :: a
      real(8), dimension(nx), intent(inout) :: b
    end subroutine
  end interface
end python module
'''

    sig_c = make_signature_c(src_c)
    equal(ref_sig_c, sig_c)
