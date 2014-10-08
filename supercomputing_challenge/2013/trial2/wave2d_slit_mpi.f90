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




SUBROUTINE periodic_x(nx, ny, f)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, ny
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f

  f(nx,:) = f(2,:)
  f(1,:) = f(nx-1,:)
END SUBROUTINE




SUBROUTINE double_slit(nx, ny, x0, width, depth, gap, c)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, ny, x0, width, depth, gap
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: c

  c(x0:x0+depth,:) = 0D0
  c(x0:x0+depth,ny/2-(gap+width)/2:ny/2+(gap+width)/2) = 0.5D0
  c(x0:x0+depth,ny/2-(gap-width)/2:ny/2+(gap-width)/2) = 0D0
END SUBROUTINE




SUBROUTINE exchange_boundary(nx, ny, f)
  IMPLICIT NONE
  INCLUDE 'mpif.h'
  INTEGER, INTENT(IN) :: nx, ny
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f
  INTEGER :: ierr, nprocs, myrank
  INTEGER :: ireq1, ireq2, ireq3, ireq4, status(MPI_STATUS_SIZE)
  INTEGER :: next_rank, prev_rank

  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)


  next_rank = myrank + 1
  prev_rank = myrank - 1

  IF (myrank == 0) THEN
    prev_rank = nprocs-1
  ELSE IF (myrank == nprocs-1) THEN
    next_rank = 0
  END IF

  CALL MPI_ISEND(f(1,ny-1), nx, MPI_REAL8, next_rank, 10, MPI_COMM_WORLD, ireq1, ierr)
  CALL MPI_ISEND(f(1,2), nx, MPI_REAL8, prev_rank, 20, MPI_COMM_WORLD, ireq2, ierr)
  CALL MPI_IRECV(f(1,ny), nx, MPI_REAL8, next_rank, 20, MPI_COMM_WORLD, ireq3, ierr)
  CALL MPI_IRECV(f(1,1), nx, MPI_REAL8, prev_rank, 10, MPI_COMM_WORLD, ireq4, ierr)

  CALL MPI_WAIT(ireq1, status, ierr)
  CALL MPI_WAIT(ireq2, status, ierr)
  CALL MPI_WAIT(ireq3, status, ierr)
  CALL MPI_WAIT(ireq4, status, ierr)
END SUBROUTINE




PROGRAM wave2d
  IMPLICIT NONE
  INCLUDE 'mpif.h'
  INTEGER, PARAMETER :: tnx=16000, tny=16000
  INTEGER, PARAMETER :: distance=600
  INTEGER, PARAMETER :: tmax=2400
  INTEGER, PARAMETER :: width=100, thick=40, gap=1000   ! slit parameters
  REAL(KIND=8), ALLOCATABLE, DIMENSION(:,:) :: c, f, g
  REAL(KIND=8), DIMENSION(tnx,tny) :: fout
  INTEGER :: nx, ny
  INTEGER :: i,j,tstep
  INTEGER :: ierr, nprocs, myrank, rank


  !----------------------------------------------------------------------------
  ! initialize the MPI environmnet
  !----------------------------------------------------------------------------
  CALL MPI_INIT(ierr)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)


  !----------------------------------------------------------------------------
  ! allocate the field arrays
  !----------------------------------------------------------------------------
  nx = 1200
  ny = 1000/nprocs + 2
  ALLOCATE(c(nx,ny))
  ALLOCATE(f(nx,ny))
  ALLOCATE(g(nx,ny))


  !----------------------------------------------------------------------------
  ! initialize coefficient and fields
  !----------------------------------------------------------------------------
  DO j=1,ny
    DO i=1,nx
      !c(i,j) = 0.25D0
      c(i,j) = 0.5D0
      f(i,j) = 0D0
      g(i,j) = 0D0
    END DO
  END DO


  !----------------------------------------------------------------------------
  ! slit structure
  !----------------------------------------------------------------------------
  IF (myrank == nprocs/2) THEN
    IF (MOD(nprocs,2) == 0) THEN
        c(:,2:2+thick) = 0D0
        c(nx/2-(gap+width)/2:nx/2+(gap+width)/2,2:2+thick) = 0.5D0
        c(nx/2-(gap-width)/2:nx/2+(gap-width)/2,2:2+thick) = 0D0
    ELSE
        c(:,ny/2:ny/2+thick) = 0D0
        c(nx/2-(gap+width)/2:nx/2+(gap+width)/2,ny/2:ny/2+thick) = 0.5D0
        c(nx/2-(gap-width)/2:nx/2+(gap-width)/2,ny/2:ny/2+thick) = 0D0
    END IF
  END IF
  

  !----------------------------------------------------------------------------
  ! main loop for the time evolution
  !----------------------------------------------------------------------------
  DO tstep=1,tmax
    !------------------------------------
    ! point source
    !------------------------------------
    !g(nx/3,ny/2) = SIN(0.02*tstep)


    !------------------------------------
    ! line source
    !------------------------------------
    !g(:,src_y0) = SIN(0.02*tstep)
    IF (MOD(nprocs,2) == 0) THEN
      IF (myrank == nprocs/2 - distance/ny - 1) THEN
        g(:,ny-(distance-distance/ny*ny)) = SIN(0.02*tstep)
      END IF
    ELSE
      IF (myrank == nprocs/2 - (distance-ny/2)/ny - 1) THEN
        g(:,ny-(distance-ny/2-distance/ny*ny)) = SIN(0.02*tstep)
      END IF
    END IF


    CALL advance_field(nx, ny, c, f, g)
    CALL periodic_x(nx, ny, f)
    CALL exchange_boundary(nx, ny, f)

    CALL advance_field(nx, ny, c, g, f)
    CALL periodic_x(nx, ny, g)
    CALL exchange_boundary(nx, ny, g)
  END DO


  !----------------------------------------------------------------------------
  ! gather fields and save as binary files
  !----------------------------------------------------------------------------
  CALL MPI_GATHER(f(1,2), nx*(ny-2), MPI_REAL8, fout, nx*(ny-2), MPI_REAL8, 0, MPI_COMM_WORLD, ierr)

  IF (myrank == 0) THEN
    OPEN(UNIT=11, FILE='field.bin', ACCESS='DIRECT', FORM='UNFORMATTED', &
        RECL=tnx*tny*8, STATUS='NEW')
    WRITE(11,rec=1) fout
    CLOSE(11)
  END IF


  !----------------------------------------------------------------------------
  ! finalize the MPI environmnet
  !----------------------------------------------------------------------------
  CALL MPI_FINALIZE(ierr)
END PROGRAM
