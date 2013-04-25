      SUBROUTINE Fib(a,n)
      INTEGER n
      REAL*8 a(n)
Cf2py intent(in),depend(a) n=len(a)      
Cf2py intent(in,out) a
      DO i=1,n
         IF (i.EQ.1) THEN
            a(i) = 0.0D0
         ELSEIF (i.EQ.2) THEN
            a(i) = 1.0D0
         ELSE
            a(i) = a(i-1) + a(i-2)
         ENDIF
      ENDDO
      END
