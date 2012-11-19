MODULE mod
  INTEGER i
  INTEGER :: x(4)
  REAL, DIMENSION(2,3) :: a
  REAL, ALLOCATABLE, DIMENSION(:,:) :: b

CONTAINS
  SUBROUTINE foo
    INTEGER k
    PRINT*, "i=", i
    PRINT*, "x=[", X, "]"
    PRINT*, "a=["
    PRINT*, "[", a(1,1), ",", a(1,2), ",", a(1,3), "]"
    PRINT*, "[", a(2,1), ",", a(2,2), ",", a(2,3), "]"
    PRINT*, "]"
    PRINT*, "Setting a(1,2)+3"
    a(1,2) = a(1,2) + 3
  END SUBROUTINE foo

  SUBROUTINE bar
    INTEGER k
    IF( ALLOCATED(b) ) THEN
      PRINT*, "b=["
      DO k = 1,size(b,1)
        PRINT*, B(k, 1:size(b,2))
      ENDDO
      PRINT*, "]"
    ELSE
      PRINT*, "b is not allocated"
    ENDIF
  END SUBROUTINE bar
END MODULE mod
