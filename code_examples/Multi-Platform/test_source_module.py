from __future__ import division
import numpy as np
import os
from numpy import pi, sqrt, sin, cos
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_get_module_f90():
    from source_module import get_module_f90

    # setup
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)


    src_f = '''
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

    mod_f = get_module_f90(src_f)
    add_f = mod_f.add

    add_f(nx, a, b, c)
    a_equal(a+b, c)




def test_get_module_c():
    from source_module import get_module_c

    # setup
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.zeros(nx)

    src_c = '''
void add(int nx, double *a, double *b, double *c) {
    int i;

    for (i=0; i<nx; i++) {
        c[i] = a[i] + b[i];
    }
}
    '''

    sig_c = '''
python module $MODNAME
  interface
    subroutine add(n,a,b,c)
      intent(c) :: add
      intent(c)    ! Adds to all following definitions
      integer, required, intent(in) :: n
      double precision, intent(in) :: a(n), b(n)
      double precision, intent(inout) :: c(n)
    end subroutine
  end interface
end python module
'''

    mod_c = get_module_c(src_c, sig_c)
    add_c = mod_c.add

    add_c(nx, a, b, c)
    a_equal(a+b, c)




def test_make_signature_c():
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
