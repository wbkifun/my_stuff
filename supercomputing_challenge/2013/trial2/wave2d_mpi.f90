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




SUBROUTINE exchange_boundary(nx, ny, f)
  IMPLICIT NONE
  INCLUDE 'mpif.h'
  INTEGER, INTENT(IN) :: nx, ny
  REAL(KIND=8), DIMENSION(nx,ny), INTENT(INOUT) :: f
  INTEGER :: ierr, nprocs, myrank
  INTEGER :: ireq1, ireq2, ireq3, ireq4, status(MPI_STATUS_SIZE)

  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)


  IF (myrank == 0) THEN
    CALL MPI_ISEND(f(1,ny-1), nx, MPI_REAL8, 1, 10, MPI_COMM_WORLD, ireq1, ierr)
    CALL MPI_IRECV(f(1,ny), nx, MPI_REAL8, 1, 20, MPI_COMM_WORLD, ireq2, ierr)

  ELSE IF (myrank == nprocs-1) THEN
    CALL MPI_ISEND(f(1,2), nx, MPI_REAL8, nprocs-2, 20, MPI_COMM_WORLD, ireq1, ierr)
    CALL MPI_IRECV(f(1,1), nx, MPI_REAL8, nprocs-2, 10, MPI_COMM_WORLD, ireq2, ierr)

  ELSE
    CALL MPI_ISEND(f(1,ny-1), nx, MPI_REAL8, myrank+1, 10, MPI_COMM_WORLD, ireq1, ierr)
    CALL MPI_ISEND(f(1,2), nx, MPI_REAL8, myrank-1, 20, MPI_COMM_WORLD, ireq2, ierr)
    CALL MPI_IRECV(f(1,ny), nx, MPI_REAL8, myrank+1, 20, MPI_COMM_WORLD, ireq3, ierr)
    CALL MPI_IRECV(f(1,1), nx, MPI_REAL8, myrank-1, 10, MPI_COMM_WORLD, ireq4, ierr)

    CALL MPI_WAIT(ireq3, status, ierr)
    CALL MPI_WAIT(ireq4, status, ierr)
  END IF

  CALL MPI_WAIT(ireq1, status, ierr)
  CALL MPI_WAIT(ireq2, status, ierr)
END SUBROUTINE




PROGRAM wave2d
  IMPLICIT NONE
  INCLUDE 'mpif.h'
  INTEGER, PARAMETER :: tnx=1200, tny=1000
  INTEGER, PARAMETER :: tmax=500
  REAL(KIND=8), ALLOCATABLE, DIMENSION(:,:) :: c, f, g
  REAL(KIND=8), DIMENSION(tnx,tny) :: fout
  INTEGER :: nx, ny
  INTEGER :: i,j,tn
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
      c(i,j) = 0.25D0
      f(i,j) = 0D0
      g(i,j) = 0D0
    END DO
  END DO
  

  !----------------------------------------------------------------------------
  ! main loop for the time evolution
  !----------------------------------------------------------------------------
  DO tn=1,tmax
    IF (myrank == 0) THEN
      ! update the source at a point
      g(nx/3,ny-1) = g(nx/3,ny-1) + SIN(0.1*tn)
    END IF

    CALL advance_field(nx, ny, c, f, g)
    CALL exchange_boundary(nx, ny, f)
    CALL advance_field(nx, ny, c, g, f)
    CALL exchange_boundary(nx, ny, g)
  END DO


  !----------------------------------------------------------------------------
  ! gather fields and save as binary files
  !----------------------------------------------------------------------------
  CALL MPI_GATHER(f(1,2), nx*(ny-2), MPI_REAL8, fout, nx*(ny-2), MPI_REAL8, 0, MPI_COMM_WORLD, ierr)

  IF (myrank == 0) THEN
    OPEN(UNIT=11, FILE='f.bin', ACCESS='DIRECT', FORM='UNFORMATTED', &
        RECL=tnx*tny*8, STATUS='NEW')
    WRITE(11,rec=1) fout
    CLOSE(11)
  END IF


  !----------------------------------------------------------------------------
  ! finalize the MPI environmnet
  !----------------------------------------------------------------------------
  CALL MPI_FINALIZE(ierr)

END PROGRAM
