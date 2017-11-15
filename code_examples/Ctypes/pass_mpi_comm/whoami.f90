MODULE whoami
  use mpi
  IMPLICIT NONE

CONTAINS
  SUBROUTINE print_myrank(comm)
    INTEGER, INTENT(IN) :: comm
    INTEGER :: nproc, myrank, ierr

    call mpi_comm_size(comm, nproc, ierr)
    call mpi_comm_rank(comm, myrank, ierr)

    print *, 'nproc=', nproc, 'myrank=', myrank
  END SUBROUTINE
END MODULE
