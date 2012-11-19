MODULE sumavgmod
  USE summod
  USE avgmod

CONTAINS
  SUBROUTINE sumavg(a,b,c)
    IMPLICIT NONE
    REAL, INTENT(IN) :: a, b, c

    PRINT*, 'sum=', sum(a,b,c)
    PRINT*, 'avg=', avg(a,b,c)

  END SUBROUTINE
END MODULE

