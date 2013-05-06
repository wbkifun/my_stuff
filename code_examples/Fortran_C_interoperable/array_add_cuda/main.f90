PROGRAM main
  USE ISO_C_BINDING
  IMPLICIT NONE

  INTERFACE
    SUBROUTINE my_routine(nx, ny, aa, bb, cc) BIND(C,name='add')
      IMPORT :: C_INT, C_PTR
      INTEGER(C_INT), VALUE :: nx, ny
      TYPE(C_PTR), VALUE :: aa, bb, cc
    END SUBROUTINE
  END INTERFACE
  

  DOUBLE PRECISION, DIMENSION(4,3), TARGET :: aa, bb, cc
  TYPE(C_PTR) :: ap, bp, cp
  
  ap = c_loc(aa(1,1))
  bp = c_loc(bb(1,1))
  cp = c_loc(cc(1,1))

  WRITE(*,'(a,z20,a,z20,a,z20)') &
          '#fortran address: aa', ap, ', bb', bp, ', cc', cp

  aa = 1.5D0
  bb = 0.7D0
  cc = 0.0D0
  CALL my_routine(4,3,ap,bp,cp)

  aa(1,1)=11.0D0
  aa(2,1)=12.0D0
  aa(3,1)=13.0D0
  CALL my_routine(4,3,ap,bp,cp)

END PROGRAM main
