SUBROUTINE Fib(a,n)
INTEGER :: n
REAL(KIND=8), DIMENSION(n) :: a
cf2py intent(in),depend(a) :: n=len(a)      
cf2py intent(in,out) :: a
DO i=1,n
   IF (i.EQ.1) THEN
      a(i) = 0.0D0
   ELSE IF (i.EQ.2) THEN
      a(i) = 1.0D0
   ELSE
      a(i) = a(i-1) + a(i-2)
   END IF
END DO
END SUBROUTINE Fib
