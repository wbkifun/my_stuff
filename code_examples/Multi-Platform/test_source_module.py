from __future__ import division
import numpy as np
import os
from numpy import pi, sqrt, sin, cos
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_



from source_module import get_module_f90, get_module_c



def test_get_module_f90():
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

    sig_f = '''
python module $MODNAME
  interface
    subroutine add(n,a,b,c)
      integer, required, intent(in) :: n
      double precision, intent(in) :: a(n), b(n)
      double precision, intent(inout) :: c(n)
    end subroutine
  end interface
end python module
'''

    mod_f = get_module_f90(src_f, sig_f)
    add_f = mod_f.add

    add_f(nx, a, b, c)
    a_equal(a+b, c)




def test_get_module_c():
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
