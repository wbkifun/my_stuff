SUBROUTINE advance_field(c, f, g, nx, ny)
  IMPLICIT NONE
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(IN)    :: c, g
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f
  INTEGER, INTENT(IN) :: nx, ny

  INTEGER :: i,j

  DO j=2,ny-1
    DO i=2,nx-1
      f(i,j) = c(i,j)*(g(i+1,j) + g(i-1,j) + g(i,j+1) + g(i,j-1) - 4*g(i,j)) &
          + 2*g(i,j) - f(i,j)
    END DO
  END DO
END SUBROUTINE




SUBROUTINE exchange_boundary(f, nx, ny)
  INCLUDE 'mpif.h'
  IMPLICIT NONE
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f
  INTEGER :: ierr, nprocs, myrank
  INTEGER :: ireq1, ireq2, ireq3, ireq4, status(MPI_STATUS_SIZE)

  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)


  IF (myrank == 0) THEN
    CALL MPI_ISEND(f(nx-1,1), ny, MPI_REAL8, 1, 10, MPI_COMM_WORLD, ireq1, ierr)
    CALL MPI_IRECV(f(nx,1), ny, MPI_REAL8, 1, 20, MPI_COMM_WORLD, ireq2, ierr)

  ELSE IF (myrank == nprocs-1) THEN
    CALL MPI_ISEND(f(2,1), ny, MPI_DOUBLE, nprocs-2, 20, MPI_COMM_WORLD, ireq1, ierr)
    CALL MPI_IRECV(f(1,1), ny, MPI_REAL8, nprocs-2, 10, MPI_COMM_WORLD, ireq2, ierr)

  ELSE
    CALL MPI_ISEND(f(nx-1,1), ny, MPI_DOUBLE, myrank+1, 10, MPI_COMM_WORLD, ireq1, ierr)
    CALL MPI_ISEND(f(2,1), ny, MPI_DOUBLE, myrank-1, 20, MPI_COMM_WORLD, ireq2, ierr)
    CALL MPI_IRECV(f(nx,1), ny, MPI_REAL8, myrank+1, 20, MPI_COMM_WORLD, ireq3, ierr)
    CALL MPI_IRECV(f(1,1), ny, MPI_REAL8, myrank-1, 10, MPI_COMM_WORLD, ireq4, ierr)

    CALL MPI_WAIT(ireq3, status, ierr)
    CALL MPI_WAIT(ireq4, status, ierr)
  END IF

  CALL MPI_WAIT(ireq1, status, ierr)
  CALL MPI_WAIT(ireq2, status, ierr)
END SUBROUTINE




PROGRAM wave2d
  INCLUDE 'mpif.h'
  IMPLICIT NONE
  INTEGER, PARAMETER :: tnx=1200, tny=1000
  INTEGER, PARAMETER :: tmax=500
  REAL(KIND=8), ALLOCATABLE, DIMENSION(:,:) :: c, f, g
  INTEGER :: nx, ny
  INTEGER :: i,j,tn
  INTEGER :: ierr, nprocs, myrank, request


  !----------------------------------------------------------------------------
  ! initialize the MPI environmnet
  !----------------------------------------------------------------------------
  CALL MPI_INIT(ierr)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)


  !----------------------------------------------------------------------------
  ! allocate the field arrays
  !----------------------------------------------------------------------------
  nx = 1200/nprocs + 2
  ny = 1000
  ALLOCATE(c(nx,ny))
  ALLOCATE(f(nx,ny))
  ALLOCATE(g(nx,ny))


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
    CALL exchange_boundary(f, nx, ny)
    CALL advance_field(c, g, f, nx, ny)
    CALL exchange_boundary(g, nx, ny)
  END DO


  !----------------------------------------------------------------------------
  ! save as binary files
  !----------------------------------------------------------------------------
  OPEN(UNIT=11, FILE='f.bin', ACCESS='DIRECT', FORM='UNFORMATTED', &
      RECL=nx*ny*8, STATUS='NEW')
  WRITE(11,rec=1) f
  CLOSE(11)


  !----------------------------------------------------------------------------
  ! finalize the MPI environmnet
  !----------------------------------------------------------------------------
  CALL MPI_FINALIZE(ierr)

END PROGRAM
