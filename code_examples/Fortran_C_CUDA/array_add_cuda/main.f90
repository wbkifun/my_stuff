PROGRAM main
  USE ISO_C_BINDING
  IMPLICIT NONE

  INTERFACE
    SUBROUTINE my_routine(nx, ny, a, b, c) BIND(C,name='add')
      IMPORT :: C_INT, C_PTR
      INTEGER(C_INT), VALUE :: nx, ny
      TYPE(C_PTR), VALUE :: a, b, c
    END SUBROUTINE
  END INTERFACE
  

  DOUBLE PRECISION, DIMENSION(4,3), TARGET :: a, b, c
  TYPE(C_PTR) :: ap, bp, cp
  
  ap = c_loc(a(1,1))
  bp = c_loc(b(1,1))
  cp = c_loc(c(1,1))

  WRITE(*,'(a,z20,a,z20,a,z20)') &
          '#fortran address: a', ap, ', b', bp, ', c', cp

  a = 1.5D0
  b = 0.7D0
  c = 0.0D0
  CALL my_routine(4,3,ap,bp,cp)

  a(1,1)=11.0D0
  a(2,1)=12.0D0
  a(3,1)=13.0D0
  CALL my_routine(4,3,ap,bp,cp)

END PROGRAM main
