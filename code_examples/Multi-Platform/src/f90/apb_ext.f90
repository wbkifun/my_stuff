#include "param1.f90.h"
#include "param2.f90.h"

MODULE apb_ext
  IMPLICIT NONE
  PRIVATE

  PUBLIC :: bpc

CONTAINS
FUNCTION bpc(b, c) RESULT(ret)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: b, c
  REAL(8) :: ret
                
  ret = LLL*b + MM*c
END FUNCTION bpc

END MODULE apb_ext
