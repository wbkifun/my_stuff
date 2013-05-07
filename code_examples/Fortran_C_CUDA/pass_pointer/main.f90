PROGRAM main
  USE ISO_C_BINDING
  IMPLICIT NONE

  INTERFACE
    SUBROUTINE my_routine(row, col, ptr) BIND(C,name='cfunc')
      IMPORT :: C_INT, C_PTR
      INTEGER(C_INT), VALUE :: row, col
      TYPE(C_PTR), VALUE :: ptr
    END SUBROUTINE
  END INTERFACE
  

  !REAL(C_DOUBLE), ALLOCATABLE, TARGET :: xyz(:,:)
  !REAL(C_DOUBLE), TARGET :: abc(4,3)
  DOUBLE PRECISION, ALLOCATABLE, TARGET :: xyz(:,:)
  DOUBLE PRECISION, TARGET :: abc(4,3)
  TYPE(C_PTR) :: cptr
  
  ALLOCATE(xyz(4,3))
  cptr = c_loc(xyz(1,1))
  WRITE(*,'(a,z20)') '#fortran address:', c_loc(xyz(1,1))
  
  xyz=-1.0
  CALL my_routine(4,3,cptr)

  xyz(1,1)=1.0
  xyz(2,1)=2.0
  xyz(3,1)=3.0
  CALL my_routine(4,3,cptr)

  DEALLOCATE(xyz)

  WRITE(*,'(a,z20)') '#fortran address:', c_loc(abc(1,1)) 
  cptr = c_loc(abc(1,1))
  abc=-5.0
  CALL my_routine(4,3,cptr)
  abc(1,1)=11.0
  abc(2,1)=12.0
  abc(3,1)=13.0
  CALL my_routine(4,3,cptr)

END PROGRAM main
