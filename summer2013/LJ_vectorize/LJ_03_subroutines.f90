SUBROUTINE force(n, lje, ljs, x, y, z, fx, fy, fz)
INTEGER, INTENT(IN) :: n
DOUBLE PRECISION, INTENT(IN) :: lje, ljs
DOUBLE PRECISION, DIMENSION(n), INTENT(IN) :: x, y, z
DOUBLE PRECISION, DIMENSION(n), INTENT(INOUT) :: fx, fy, fz

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



SUBROUTINE output_energy(n, tstep, em, lje, ljs, x, y, z, vx, vy, vz, energy)
INTEGER, INTENT(IN) :: n, tstep
DOUBLE PRECISION, INTENT(IN) :: em, lje, ljs
DOUBLE PRECISION, DIMENSION(n), INTENT(IN) :: x, y, z
DOUBLE PRECISION, DIMENSION(n), INTENT(IN) :: vx, vy, vz
DOUBLE PRECISION, DIMENSION(3), INTENT(INOUT) :: energy

INTEGER :: i, j
DOUBLE PRECISION :: r, ke, pe, te
DOUBLE PRECISION :: dx, dy, dz


! kinetic energy
ke = 0.0d0
DO i=0,n-1
  ke = ke + vx(i)**2 + vy(i)**2 + vz(i)**2
END DO
ke = 0.5 * em * ke

! potential energy
pe = 0.0d0
DO j=0,n-1
  DO i=0,n-1
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
energy(1) = ke
energy(2) = pe
energy(3) = te

!PRINT *, tstep, ke, pe, te 
END SUBROUTINE
