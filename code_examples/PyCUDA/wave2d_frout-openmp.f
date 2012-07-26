      SUBROUTINE update(nx, ny, c, f, g)
      INTEGER nx, ny
      REAL c(0:nx-1,0:ny-1), f(0:nx-1,0:ny-1), g(0:nx-1,0:ny-1)
Cf2py intent(in,hide), DEPEND(f) nx=shape(f,0), ny=shape(f,1)
Cf2py intent(in) :: c
Cf2py intent(in,out) :: f, g

      INTEGER :: i, j
C$OMP PARALLEL SHARED(nx, ny, c, f, g) PRIVATE(i, j)
C$OMP DO SCHEDULE(guided)
      DO j=1,ny-2
         DO i = 1, nx-2
            f(i,j) = c(i,j)*(g(i,j+1) + g(i,j-1) + g(i+1,j) + g(i-1,j)
     +      - 4*g(i,j)) + 2*g(i,j) - f(i,j)
         END DO
      END DO
C$OMP END DO NOWAIT
C$OMP END PARALLEL
      END SUBROUTINE update
