SUBROUTINE advance_field(c, f, g, nx, ny)
  IMPLICIT NONE
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(IN)    :: c, g
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f
  INTEGER, INTENT(IN) :: nx, ny

  INTEGER :: i,j

  !$OMP PARALLEL DO
  DO j=2,ny-1
    DO i=2,nx-1
      f(i,j) = c(i,j)*(g(i+1,j) + g(i-1,j) + g(i,j+1) + g(i,j-1) - 4*g(i,j)) &
          + 2*g(i,j) - f(i,j)
    END DO
  END DO
  !$OMP END PARALLEL DO
  
END SUBROUTINE



PROGRAM wave2d
  IMPLICIT NONE
  INTEGER, PARAMETER :: nx=1200, ny=1000
  INTEGER, PARAMETER :: tmax=500
  REAL(KIND=8), DIMENSION(nx,ny) :: c, f, g
  INTEGER :: i,j,tn


  !----------------------------------------------------------------------------
  ! initialize coefficient and fields
  !----------------------------------------------------------------------------
  DO j=1,ny
    DO i=1,nx
      c(i,j) = 0.25D0
      f(i,j) = 0D0
      g(i,j) = 0D0
    END DO
  END DO
  

  !----------------------------------------------------------------------------
  ! main loop for the time evolution
  !----------------------------------------------------------------------------
  DO tn=1,tmax
    ! update the source at a point
    g(nx/3,ny/2) = g(nx/3,ny/2) + SIN(0.1*tn)

    CALL advance_field(c, f, g, nx, ny)
    CALL advance_field(c, g, f, nx, ny)
  END DO


  !----------------------------------------------------------------------------
  ! save as binary files
  !----------------------------------------------------------------------------
  OPEN(UNIT=11, FILE='f.bin', ACCESS='DIRECT', FORM='UNFORMATTED', &
      RECL=nx*ny*8, STATUS='NEW')
  WRITE(11,rec=1) f
  CLOSE(11)
END PROGRAM
