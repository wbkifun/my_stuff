PROGRAM main
  USE ISO_C_BINDING
  IMPLICIT NONE

  INTERFACE
    SUBROUTINE my_routine(r, c, p) BIND(c,name='cfunc')
      IMPORT :: c_int, c_ptr
      INTEGER(c_int), value :: r, c
      TYPE(c_ptr), value :: p
    END SUBROUTINE
  END INTERFACE
  

  REAL(C_DOUBLE), ALLOCATABLE, TARGET :: xyz(:,:)
  REAL(C_DOUBLE), TARGET :: abc(7,3)
  TYPE(c_ptr) :: cptr
  
  ALLOCATE(xyz(7,3))
  cptr = c_loc(xyz(1,1))
  WRITE(*,'(a,z20)') '#fortran address:', c_loc(xyz(1,1))
  
  xyz=-1.0
  CALL my_routine(7,3,cptr)

  xyz(1,1)=1.0
  xyz(2,1)=2.0
  xyz(3,1)=3.0
  CALL my_routine(7,3,cptr)

  DEALLOCATE(xyz)

  WRITE(*,'(a,z20)') '#fortran address:', c_loc(abc(1,1)) 
  cptr = c_loc(abc(1,1))
  abc=-5.0
  CALL my_routine(7,3,cptr)
  abc(1,1)=11.0
  abc(2,1)=12.0
  abc(3,1)=13.0
  CALL my_routine(7,3,cptr)

END PROGRAM main
