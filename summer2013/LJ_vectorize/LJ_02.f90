PROGRAM LJ
IMPLICIT NONE

INTEGER, PARAMETER :: n=500    ! Number of atoms, molecules
INTEGER, PARAMETER :: mt=1000   ! max time steps
INTEGER, PARAMETER :: dt=10     ! time interval (a.u.)
INTEGER, PARAMETER :: domain=600    ! domain size (a.u.)
DOUBLE PRECISION, PARAMETER :: ms=0.00001    ! max speed (a.u.)
DOUBLE PRECISION, PARAMETER :: em=1822.88839*28.0134    ! effective mass of N2
DOUBLE PRECISION, PARAMETER :: lje=0.000313202   ! Lennard-Jones epsilon of N2
DOUBLE PRECISION, PARAMETER :: ljs=6.908841465   ! Lennard-Jones sigma of N2

DOUBLE PRECISION, DIMENSION(3*n) :: x, v, f, tmp
INTEGER tstep


! Initialize
CALL RANDOM_NUMBER(tmp)
x = domain * tmp
v = ms * (tmp - 0.5)


! Dynamics
DO tstep=1,mt
  CALL force(n, lje, ljs, x, f)
  CALL solve(n, dt, em, x, v, f)

  !CALL output_energy(tstep, n, em, lje, ljs, x, v)
END DO



CONTAINS
  SUBROUTINE force(n, lje, ljs, x, f)
    INTEGER, INTENT(IN) :: n
    DOUBLE PRECISION, INTENT(IN) :: lje, ljs
    DOUBLE PRECISION, DIMENSION(3*n), INTENT(IN) :: x
    DOUBLE PRECISION, DIMENSION(3*n), INTENT(OUT) :: f

    INTEGER :: i, j
    DOUBLE PRECISION :: r, r2, pe
    DOUBLE PRECISION, DIMENSION(3) :: d, t

    DO j=0,n-1
      t = 0

      DO i=0,n-1
        IF (i==j) CYCLE
        !d = x(3*i+1:3*i+3) - x(3*j+1:3*j+3) 
        d(1) = x(3*i+1) - x(3*j+1) 
        d(2) = x(3*i+2) - x(3*j+2) 
        d(3) = x(3*i+3) - x(3*j+3) 
        r = SQRT(d(1)**2 + d(2)**2 + d(3)**2)
        r2 = r**2

        pe = 2*(ljs/r)**12 - (ljs/r)**6
        !t = t + pe*d/r2
        t(1) = t(1) + pe*d(1)/r2
        t(2) = t(2) + pe*d(2)/r2
        t(3) = t(3) + pe*d(3)/r2
      END DO

      f(3*j+1:3*j+3) = -24 * lje * t
    END DO
  END SUBROUTINE


  SUBROUTINE solve(n, dt, em, x, v, f)
    INTEGER, INTENT(IN) :: n, dt
    DOUBLE PRECISION, INTENT(IN) :: em
    DOUBLE PRECISION, DIMENSION(3*n), INTENT(INOUT) :: x, v
    DOUBLE PRECISION, DIMENSION(3*n), INTENT(IN) :: f

    x = x + v*dt
    v = v + f*dt/em
  END SUBROUTINE


  SUBROUTINE output_energy(tstep, n, em, lje, ljs, x, v)
    INTEGER, INTENT(IN) :: tstep, n
    DOUBLE PRECISION, INTENT(IN) :: em, lje, ljs
    DOUBLE PRECISION, DIMENSION(3*n), INTENT(IN) :: x, v

    INTEGER :: i, j
    DOUBLE PRECISION, DIMENSION(3) :: d
    DOUBLE PRECISION :: r, ke, pe, te


    ! kinetic energy
    ke = 0
    DO i=0,n-1
      ke = ke + v(3*i+1)**2 + v(3*i+2)**2 + v(3*i+3)**2
    END DO
    ke = 0.5 * em * ke

    ! potential energy
    pe = 0
    DO j=0,n-1
      DO i=0,n-1
        IF (i==j) CYCLE

        d = x(3*i+1:3*i+3) - x(3*j+1:3*j+3) 
        r = SQRT(d(1)**2 + d(2)**2 + d(3)**2)
        pe = pe + 4*lje*((ljs/r)**12 - (ljs/r)**6)
      END DO
    END DO
    pe = 0.5 * pe

    te = ke + pe

    PRINT *, tstep, ke, pe, te 
  END SUBROUTINE
END PROGRAM
