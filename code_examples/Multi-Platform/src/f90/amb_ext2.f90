MODULE amb_ext2
  IMPLICIT NONE
  PRIVATE

  PUBLIC :: mc

CONTAINS
FUNCTION mc(m, c) RESULT(ret)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: m
  REAL(8), INTENT(IN) :: c
  REAL(8) :: ret
                
  ret = m*c
END FUNCTION mc

END MODULE amb_ext2
