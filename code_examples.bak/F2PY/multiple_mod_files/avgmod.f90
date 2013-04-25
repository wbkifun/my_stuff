MODULE avgmod
  USE summod

CONTAINS
  REAL FUNCTION avg(a,b,c)
    IMPLICIT NONE
    REAL, INTENT(IN) :: a, b, c

    avg = sum(a,b,c) / 3.0

  END FUNCTION
END MODULE
