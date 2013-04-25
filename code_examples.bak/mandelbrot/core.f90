SUBROUTINE diverge(c, z, numdiv, maxiter, nx, ny)
  IMPLICIT NONE
  COMPLEX*16, DIMENSION(nx,ny), INTENT(IN)    :: c
  COMPLEX*16, DIMENSION(nx,ny), INTENT(INOUT) :: z
  INTEGER, DIMENSION(nx,ny), INTENT(INOUT)    :: numdiv
  INTEGER, INTENT(IN) :: maxiter, nx, ny

  INTEGER :: i,j,n

  DO n=1,maxiter
    DO j=1,ny
      DO i=1,nx
        z(i,j) = z(i,j)**2 + c(i,j)
        IF (abs(z(i,j)) > 2) THEN
          z(i,j) = 2
          IF (numdiv(i,j) == maxiter) THEN
            numdiv(i,j) = n
          END IF
        END IF
      END DO
    END DO
  END DO
END SUBROUTINE
