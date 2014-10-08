SUBROUTINE advance_field(nx, ny, c, f, g)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, ny
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(IN)    :: c, g
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f

  INTEGER :: i,j

  DO j=2,ny-1
    DO i=2,nx-1
      f(i,j) = c(i,j)*(g(i+1,j) + g(i-1,j) + g(i,j+1) + g(i,j-1) - 4*g(i,j)) &
          + 2*g(i,j) - f(i,j)
    END DO
  END DO
END SUBROUTINE




SUBROUTINE point_src(nx, ny, nx0, ny0, tstep, f)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, ny, nx0, ny0, tstep
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f

  f(nx0,ny0) = f(nx0,ny0) + SIN(0.1*tstep)
END SUBROUTINE




SUBROUTINE line_src_x_direction(nx, ny, x0, tstep, f)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, ny, x0, tstep
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f

  f(x0,:) = SIN(0.1*tstep)
END SUBROUTINE




SUBROUTINE periodic_y(nx, ny, f)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, ny
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f

  f(:,ny) = f(:,2)
  f(:,1) = f(:,ny-1)
END SUBROUTINE




SUBROUTINE single_slit(nx, ny, x0, width, depth, c)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, ny, x0, width, depth
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: c

  c(x0:x0+depth,:) = 0D0
  c(x0:x0+depth,ny/2-width/2:ny/2+width/2) = 0.25D0
END SUBROUTINE




SUBROUTINE double_slit(nx, ny, x0, width, depth, distance, c)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, ny, x0, width, depth, distance
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: c

  c(x0:x0+depth,:) = 0D0
  c(x0:x0+depth,ny/2-(distance+width)/2:ny/2+(distance+width)/2) = 0.25D0
  c(x0:x0+depth,ny/2-(distance-width)/2:ny/2+(distance-width)/2) = 0D0
END SUBROUTINE




PROGRAM wave2d
  IMPLICIT NONE
  INTEGER, PARAMETER :: nx=1200, ny=1000
  INTEGER, PARAMETER :: tmax=1500
  INTEGER, PARAMETER :: x0=2*nx/3, width=10, depth=5, distance=100   ! slit
  REAL(KIND=8), DIMENSION(nx,ny) :: c, f, g
  INTEGER :: i,j,tstep


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

  !CALL single_slit(nx, ny, x0, width, depth, c)
  CALL double_slit(nx, ny, x0, width, depth, distance, c)
  

  !----------------------------------------------------------------------------
  ! main loop for the time evolution
  !----------------------------------------------------------------------------
  DO tstep=1,tmax,2
    !CALL point_src(nx, ny, nx/3, ny/2, tstep, g)
    CALL line_src_x_direction(nx, ny, nx/3, tstep, g)
    CALL periodic_y(nx, ny, g)
    CALL advance_field(nx, ny, c, f, g)

    CALL periodic_y(nx, ny, f)
    CALL advance_field(nx, ny, c, g, f)
  END DO


  !----------------------------------------------------------------------------
  ! save as binary files
  !----------------------------------------------------------------------------
  OPEN(UNIT=11, FILE='f.bin', ACCESS='DIRECT', FORM='UNFORMATTED', &
      RECL=nx*ny*8, STATUS='NEW')
  WRITE(11,rec=1) f
  CLOSE(11)
END PROGRAM
