MODULE TestMod
  IMPLICIT NONE
  INTEGER, PARAMETER :: nx=3

  
  CONTAINS

  SUBROUTINE testsub()
    REAL(8), DIMENSION(nx,nx) :: arr
    INTEGER :: i,j

    DO j=1,nx
      DO i=1,nx
        arr(i,j) = i+j
      END DO
    END DO

    PRINT *, nx
  END SUBROUTINE testsub


  SUBROUTINE testsub2(arr, nx)
    INTEGER, INTENT(in) :: nx
    REAL(8), INTENT(inout), DIMENSION(nx,nx) :: arr
    INTEGER :: i,j

    DO j=1,nx
      DO i=1,nx
        arr(i,j) = i+j
      END DO
    END DO

    PRINT *, nx
  END SUBROUTINE testsub2


  FUNCTION testfunc(b) result (ret)
    INTEGER, INTENT(IN) :: b
    INTEGER :: ret
    ret = nx + b
  END FUNCTION testfunc
END MODULE TestMod


!PROGRAM main
!  USE TestMod, ONLY : nx, testsub, testfunc
!
!  PRINT *, nx
!  CALL testsub
!  PRINT *, testfunc(2)
!END PROGRAM main
