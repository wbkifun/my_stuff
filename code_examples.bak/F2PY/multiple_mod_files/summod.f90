MODULE summod

CONTAINS
  REAL FUNCTION sum(a,b,c)
    IMPLICIT NONE
    REAL, INTENT(IN) :: a, b, c

    sum = a + b + c

  END FUNCTION
END MODULE
