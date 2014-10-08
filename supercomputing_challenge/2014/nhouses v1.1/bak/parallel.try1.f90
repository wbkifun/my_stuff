MODULE mod_data
  INCLUDE 'mpif.h'
  INTEGER, PARAMETER :: nx=14
  INTEGER, PARAMETER :: granularity=5
  INTEGER, PARAMETER :: master=0
  INTEGER :: nprocs, myrank

  TYPE job
    LOGICAL :: working      ! true:working, false:quit
    INTEGER, DIMENSION(granularity) :: site
  END TYPE

  TYPE job_msg
    INTEGER :: solutions_found
    INTEGER :: origin
  END TYPE
END MODULE mod_data



MODULE mod_interface
  INTERFACE
    FUNCTION master_build_house(nx, column, site)
      INTEGER :: master_build_house
      INTEGER, INTENT(IN) :: nx, column
      INTEGER, DIMENSION(nx), INTENT(INOUT) :: site
    END FUNCTION

    FUNCTION worker_build_house(nx, column, site)
      INTEGER :: worker_build_house
      INTEGER, INTENT(IN) :: nx, column
      INTEGER, DIMENSION(nx), INTENT(INOUT) :: site
    END FUNCTION

    FUNCTION send_job_worker(site)
      USE mod_data, ONLY : nx
      INTEGER :: send_job_worker
      INTEGER, DIMENSION(nx), INTENT(INOUT) :: site
    END FUNCTION

    FUNCTION wait_remaining_results()
      INTEGER :: wait_remaining_results
    END FUNCTION

    SUBROUTINE worker()
    END SUBROUTINE
  END INTERFACE
END MODULE mod_interface




PROGRAM houses
  USE mod_data
  USE mod_interface
  IMPLICIT NONE
  INTEGER :: num_sol
  INTEGER, DIMENSION(nx) :: site
  INTEGER :: ierr

  CALL MPI_INIT(ierr)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)


  IF (myrank == master) THEN
    PRINT *, 'nx=', nx
    num_sol = master_build_house(nx, 1, site)
    !num_sol = num_sol + wait_remaining_results()
    PRINT *, 'num_sol=', num_sol
  ELSE
    CALL worker()
  END IF

  CALL MPI_FINALIZE(ierr)
END PROGRAM




RECURSIVE FUNCTION master_build_house(nx, column, site) RESULT(num_sol)
  USE mod_interface, ONLY : send_job_worker
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, column
  INTEGER, DIMENSION(nx), INTENT(INOUT) :: site
  INTEGER :: num_sol
  INTEGER :: i,j
  LOGICAL :: is_sol

  num_sol = 0

  ! Try to build a house in each line of column
  DO i=1,nx
    site(column) = i

    ! Check if this placement is still a solution
    is_sol = .TRUE.
    DO j=column-1,1,-1
      IF (site(column) == site(j) .OR. &
          site(column) == site(j)-(column-j) .OR. &
          site(column) == site(j)+(column-j)) THEN
        is_sol = .FALSE.
        EXIT
      END IF
    END DO


    IF (is_sol) THEN
      IF (column == nx) THEN
        ! If this is the last level (granularity of the job),
        ! send a next job to a worker
        num_sol = num_sol + send_job_worker(site)
      ELSE
        ! The placement is not complete.
        ! Try to place the house on the next column
        num_sol = num_sol + master_build_house(nx, column+1, site)
      END IF
    END IF
  END DO
END FUNCTION



RECURSIVE FUNCTION worker_build_house(nx, column, site) RESULT(num_sol)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx, column
  INTEGER, DIMENSION(nx), INTENT(INOUT) :: site
  INTEGER :: num_sol
  INTEGER :: i,j
  LOGICAL :: is_sol

  num_sol = 0

  ! Try to build a house in each line of column
  DO i=1,nx
    site(column) = i

    ! Check if this placement is still a solution
    is_sol = .TRUE.
    DO j=column-1,1,-1
      IF (site(column) == site(j) .OR. &
          site(column) == site(j)-(column-j) .OR. &
          site(column) == site(j)+(column-j)) THEN
        is_sol = .FALSE.
        EXIT
      END IF
    END DO


    IF (is_sol) THEN
      IF (column == nx) THEN
        ! If this is the last column, printout the solution
        num_sol = num_sol + 1
      ELSE
        ! The placement is not complete.
        ! Try to place the house on the next column
        num_sol = num_sol + worker_build_house(nx, column+1, site)
      END IF
    END IF
  END DO
END FUNCTION




FUNCTION send_job_worker(site) RESULT(num_sol)
  USE mod_data

  IMPLICIT NONE
  INTEGER, DIMENSION(nx), INTENT(INOUT) :: site
  INTEGER :: num_sol
  INTEGER :: i
  INTEGER :: stat(MPI_STATUS_SIZE), ierr
  TYPE(job) :: todo
  TYPE(job_msg) :: msg

  ! Set the job
  todo%working = .TRUE.

  DO i=1,granularity
    todo%site(i) = site(i)
  END DO

  ! Recieve the last result from a worker
  print *, SIZEOF(msg)
  CALL MPI_RECV(msg, SIZEOF(msg), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, &
                MPI_COMM_WORLD, stat, ierr)

  num_sol = msg%solutions_found

  ! Send the new job to the worker
  CALL MPI_SEND(todo, SIZEOF(todo), MPI_BYTE, msg%origin, 0, MPI_COMM_WORLD)
END FUNCTION




FUNCTION wait_remaining_results() RESULT(num_sol)
  ! Wait for remaining results, sending a quit whenever a new result arrives

  USE mod_data

  IMPLICIT NONE
  INTEGER :: num_sol
  INTEGER :: n_workers
  INTEGER :: i
  INTEGER :: stat(MPI_STATUS_SIZE), ierr
  TYPE(job) :: todo
  TYPE(job_msg) :: msg

  num_sol = 0
  n_workers = nprocs-1

  ! Set the job
  todo%working = .FALSE.

  DO WHILE (n_workers > 0)
    ! Receive a message from a worker
    CALL MPI_RECV(msg, SIZEOF(msg), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, &
             MPI_COMM_WORLD, stat, ierr)
    num_sol = num_sol + msg%solutions_found

    CALL MPI_SEND(todo, SIZEOF(todo), MPI_BYTE, msg%origin, 0, MPI_COMM_WORLD)

    n_workers = n_workers - 1
  END DO
END FUNCTION




SUBROUTINE worker()
  ! There is a default message named ask_job which lets a worker request a 
  ! job reporting the number of solutions found in the last iteration

  USE mod_data
  USE mod_interface, ONLY : worker_build_house

  IMPLICIT NONE
  INTEGER :: num_sol
  INTEGER :: stat(MPI_STATUS_SIZE), ierr
  TYPE(job_msg) :: msg
  TYPE(job) :: todo

  msg%origin = myrank
  msg%solutions_found = 0

  ! Request initial job
  CALL MPI_SEND(msg, SIZEOF(msg), MPI_BYTE, MASTER, 0, MPI_COMM_WORLD)

  DO WHILE (.TRUE.)
    ! Wait for a job or a quit message
   
    CALL MPI_RECV(todo, SIZEOF(todo), MPI_BYTE, MPI_ANY_SOURCE, &
                  MPI_ANY_TAG, MPI_COMM_WORLD, stat, ierr);

    IF (todo%working .EQV. .FALSE.) THEN
      EXIT
    END IF

    num_sol = worker_build_house(nx, granularity, todo%site)

    ! Ask for more work
    msg%solutions_found = num_sol

    CALL MPI_SEND(msg, SIZEOF(msg), MPI_BYTE, MASTER, 0, MPI_COMM_WORLD)
  END DO
END SUBROUTINE
