#include "param1.f90.h"

SUBROUTINE amb(nx, a, b, c)
  USE amb_ext1, ONLY : bmc

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx
  REAL(8), DIMENSION(nx), INTENT(IN) :: a, b
  REAL(8), DIMENSION(nx), INTENT(INOUT) :: c

  INTEGER :: i

  CALL bmc(nx, LLL, b, c)

  DO i=1,nx
    c(i) = KK*a(i) + c(i)
  END DO
END SUBROUTINE amb
