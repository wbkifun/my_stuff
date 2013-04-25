SUBROUTINE smoothing(data, smooth_data, n)
  IMPLICIT NONE
  REAL*8, DIMENSION(n+1), INTENT(IN) :: data
  REAL*8, DIMENSION(2:n), INTENT(OUT) :: smooth_data
  INTEGER, INTENT(IN) :: n
  INTEGER :: i

  DO i=2,n
    smooth_data(i) = 0.5D0*(data(i-1) + data(i+1))
  END DO
END SUBROUTINE
