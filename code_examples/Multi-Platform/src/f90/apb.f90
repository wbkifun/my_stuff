SUBROUTINE apb(nx, a, b, c)
  USE apb_ext, ONLY : bpc

  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx
  REAL(8), DIMENSION(nx), INTENT(IN) :: a, b
  REAL(8), DIMENSION(nx), INTENT(INOUT) :: c

  INTEGER :: i

  DO i=1,nx
    c(i) = KK*a(i) + bpc(b(i), c(i))
  END DO
END SUBROUTINE apb
