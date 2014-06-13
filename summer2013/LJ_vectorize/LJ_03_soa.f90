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

DOUBLE PRECISION, DIMENSION(n) :: x, y, z
DOUBLE PRECISION, DIMENSION(n) :: vx, vy, vz
DOUBLE PRECISION, DIMENSION(n) :: fx, fy, fz
DOUBLE PRECISION, DIMENSION(n,3) :: tmp
INTEGER :: tstep


! Initialize
CALL RANDOM_NUMBER(tmp)
x = domain * tmp(:,1)
y = domain * tmp(:,2)
z = domain * tmp(:,3)
vx = ms * (tmp(:,1) - 0.5)
vy = ms * (tmp(:,2) - 0.5)
vz = ms * (tmp(:,3) - 0.5)


! Dynamics
DO tstep=1,mt
  CALL force(n, lje, ljs, x, y, z, fx, fy, fz)
  CALL solve(n, dt, em, x, y, z, vx, vy, vz, fx, fy, fz)

  !CALL output_energy(tstep, n, em, lje, ljs, x, y, z, vx, vy, vz)
END DO

CALL output_energy(tstep, n, em, lje, ljs, x, y, z, vx, vy, vz)



CONTAINS
  SUBROUTINE force(n, lje, ljs, x, y, z, fx, fy, fz)
    INTEGER, INTENT(IN) :: n
    DOUBLE PRECISION, INTENT(IN) :: lje, ljs
    DOUBLE PRECISION, DIMENSION(n), INTENT(IN) :: x, y, z
    DOUBLE PRECISION, DIMENSION(n), INTENT(OUT) :: fx, fy, fz

    INTEGER :: i, j
    DOUBLE PRECISION :: r, r2, pe
    DOUBLE PRECISION :: dx, dy, dz, tx, ty, tz

    DO j=1,n
      tx = 0
      ty = 0
      tz = 0

      DO i=1,n
        IF (i==j) CYCLE

        dx = x(i) - x(j) 
        dy = y(i) - y(j) 
        dz = z(i) - z(j) 
        r = SQRT(dx**2 + dy**2 + dz**2)
        r2 = r**2

        pe = 2*(ljs/r)**12 - (ljs/r)**6
        tx = tx + pe*dx/r2
        ty = ty + pe*dy/r2
        tz = tz + pe*dz/r2
      END DO

      fx(j) = -24 * lje * tx
      fy(j) = -24 * lje * ty
      fz(j) = -24 * lje * tz
    END DO
  END SUBROUTINE


  SUBROUTINE solve(n, dt, em, x, y, z, vx, vy, vz, fx, fy, fz)
    INTEGER, INTENT(IN) :: n, dt
    DOUBLE PRECISION, INTENT(IN) :: em
    DOUBLE PRECISION, DIMENSION(n), INTENT(INOUT) :: x, y, z
    DOUBLE PRECISION, DIMENSION(n), INTENT(INOUT) :: vx, vy, vz
    DOUBLE PRECISION, DIMENSION(n), INTENT(IN) :: fx, fy, fz

    x = x + vx*dt
    y = y + vy*dt
    z = z + vz*dt

    vx = vx + fx*dt/em
    vy = vy + fy*dt/em
    vz = vz + fz*dt/em
  END SUBROUTINE


  SUBROUTINE output_energy(tstep, n, em, lje, ljs, x, y, z, vx, vy, vz)
    INTEGER, INTENT(IN) :: tstep, n
    DOUBLE PRECISION, INTENT(IN) :: em, lje, ljs
    DOUBLE PRECISION, DIMENSION(n), INTENT(IN) :: x, y, z
    DOUBLE PRECISION, DIMENSION(n), INTENT(IN) :: vx, vy, vz

    INTEGER :: i, j
    DOUBLE PRECISION :: r, ke, pe, te
    DOUBLE PRECISION :: dx, dy, dz


    ! kinetic energy
    ke = 0.0d0
    DO i=1,n
      ke = ke + vx(i)**2 + vy(i)**2 + vz(i)**2
    END DO
    ke = 0.5 * em * ke

    ! potential energy
    pe = 0.0d0
    DO j=1,n
      DO i=1,n
        IF (i==j) CYCLE

        dx = x(i) - x(j) 
        dy = y(i) - y(j) 
        dz = z(i) - z(j) 
        r = SQRT(dx**2 + dy**2 + dz**2)
        pe = pe + 4*lje*((ljs/r)**12 - (ljs/r)**6)
      END DO
    END DO
    pe = 0.5 * pe

    te = ke + pe

    PRINT *, tstep, ke, pe, te 
  END SUBROUTINE
END PROGRAM
