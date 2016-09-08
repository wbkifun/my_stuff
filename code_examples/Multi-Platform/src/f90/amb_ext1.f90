#include "param2.f90.h"

MODULE amb_ext1
  IMPLICIT NONE
  PRIVATE

  PUBLIC :: bmc

CONTAINS
SUBROUTINE bmc(nx, ll, b, c)
  USE amb_ext2, ONLY : mc

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx
  REAL(8), INTENT(IN) :: ll
  REAL(8), DIMENSION(nx), INTENT(IN) :: b
  REAL(8), DIMENSION(nx), INTENT(INOUT) :: c

  INTEGER :: i
                
  DO i=1,nx
    c(i) = ll*b(i) + mc(MM, c(i))
  END DO
END SUBROUTINE bmc

END MODULE amb_ext1
