SUBROUTINE advance(c, f, g, nx, ny)
  IMPLICIT NONE
  DOUBLE PRECISION, DIMENSION(nx,ny), INTENT(IN)    :: c, g
  DOUBLE PRECISION, DIMENSION(nx,ny), INTENT(INOUT) :: f
  INTEGER, INTENT(IN) :: nx, ny

  INTEGER :: i,j

  DO j=2,ny-1
    DO i=2,nx-1
      f(i,j) = c(i,j)*(g(i+1,j) + g(i-1,j) + g(i,j+1) + g(i,j-1) - 4*g(i,j)) &
          + 2*g(i,j) - f(i,j)
    END DO
  END DO
END SUBROUTINE
