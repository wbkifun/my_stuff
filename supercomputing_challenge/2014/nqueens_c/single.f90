PROGRAM n_queens
  !-----------------------------------------------------------------------
  ! Solve the NQueens problem for the <size> NxN board
  !-----------------------------------------------------------------------
  IMPLICIT NONE

  INTERFACE
    FUNCTION place_queen(column, board, size)
      INTEGER :: place_queen
      INTEGER, INTENT(IN) :: column, size
      INTEGER, DIMENSION(size), INTENT(INOUT) :: board
    END FUNCTION
  END INTERFACE


  INTEGER, PARAMETER :: size=18
  INTEGER :: n_solutions
  INTEGER, DIMENSION(size) :: board

  PRINT *, 'size=', size

  n_solutions = place_queen(1, board, size)
  PRINT *, 'n_solutions=', n_solutions
END PROGRAM




RECURSIVE FUNCTION place_queen(column, board, size) RESULT(n_solutions)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: column, size
  INTEGER, DIMENSION(size), INTENT(INOUT) :: board
  INTEGER :: n_solutions
  INTEGER :: i,j
  LOGICAL :: is_sol

  n_solutions = 0

  ! Try to place a queen in each line of <column>
  DO i=1,size
    board(column) = i

    ! Check if this board is still a solution

    is_sol = .TRUE.
    DO j=column-1,1,-1
      IF (board(column) == board(j) .OR. &
          board(column) == board(j)-(column-j) .OR. &
          board(column) == board(j)+(column-j)) THEN
        is_sol = .FALSE.
        EXIT
      END IF
    END DO


    ! It is a solution!
    IF (is_sol) THEN
      IF (column == size) THEN
        ! If this is the last column, printout the solution
        n_solutions = n_solutions + 1
      ELSE
        ! The board is not complete. Try to place the queens
        ! on the next level, using the current board
        n_solutions = n_solutions + place_queen(column+1, board, size)
      END IF
    END IF
  END DO
END FUNCTION
