SUBROUTINE saxpy(n, a, x, y)
  INTEGER,               INTENT(IN   ) :: n
  REAL(4),               INTENT(IN   ) :: a
  REAL(4), DIMENSION(n), INTENT(IN   ) :: x
  REAL(4), DIMENSION(n), INTENT(INOUT) :: y

  INTEGER :: i

  DO i=1, n
    y(i) = a*x(i) + y(i)
  END DO
END SUBROUTINE
