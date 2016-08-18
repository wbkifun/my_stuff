SUBROUTINE axpx(n, a, x, y)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a, x(2*n)
  REAL(8), INTENT(INOUT) :: y(n)

  INTEGER :: i

  DO i=1,n
    y(i) = a*(x(2*i-1) + x(2*i))
  END DO
END SUBROUTINE
