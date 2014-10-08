MODULE mod_houses
  INTEGER, PARAMETER :: nx=14

  INTERFACE
    FUNCTION build_house(nx, column, site)
      INTEGER :: build_house
      INTEGER, INTENT(IN) :: nx, column
      INTEGER, DIMENSION(nx), INTENT(INOUT) :: site
    END FUNCTION
  END INTERFACE
END MODULE



RECURSIVE FUNCTION build_house(nx, column, site) RESULT(num_sol)
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
        num_sol = num_sol + build_house(nx, column+1, site)
      END IF
    END IF
  END DO
END FUNCTION




PROGRAM houses
  USE mod_houses
  IMPLICIT NONE
  INTEGER :: num_sol
  INTEGER, DIMENSION(nx) :: site
  CHARACTER(10) :: date, time, zone
  INTEGER, DIMENSION(8) :: t1, t2
  REAL :: elapsed

  PRINT *, 'nx=', nx

  CALL DATE_AND_TIME(date, time, zone, t1)
  num_sol = build_house(nx, 1, site)
  CALL DATE_AND_TIME(date, time, zone, t2)
  !print *, t1(6), t1(7), t1(8)
  !print *, t2(6), t2(7), t2(8)
  elapsed = REAL(t2(6)-t1(6))*60. + REAL(t2(7)-t1(7)) + REAL(t2(8)-t1(8))*1e-3   ! sec

  PRINT *, 'num_sol=', num_sol
  PRINT *, 'elapsed_time=', elapsed, 'sec'
END PROGRAM




