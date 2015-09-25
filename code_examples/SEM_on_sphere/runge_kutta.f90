SUBROUTINE daxpy(n, ret, a, x, y)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a, x(n), y(n)
  REAL(8), INTENT(INOUT) :: ret(n)

  INTEGER :: i

  DO i=1,n
    ret(i) = a*x(i) + y(i)
  END DO
END SUBROUTINE




SUBROUTINE rk4sum(n, dt, k1, k2, k3, k4, ret)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: dt, k1(n), k2(n), k3(n), k4(n)
  REAL(8), INTENT(INOUT) :: ret(n)

  INTEGER :: i

  DO i=1,n
    ret(i) = ret(i) + (dt/6)*(k1(i) + 2*k2(i) + 2*k3(i) + k4(i))
  END DO
END SUBROUTINE
