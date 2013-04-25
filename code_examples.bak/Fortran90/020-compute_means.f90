PROGRAM ComputeMeans
IMPLICIT NONE

REAL :: x=1.0, y=2.0, z=3.0
REAL :: arith_mean, geo_mean, harm_mean

WRITE(*,*) 'Data items: ', x, y, z
WRITE(*,*) 

arith_mean = (x + y + z)/3.0
geo_mean = (x * y * z)**(1.0/3.0)
harm_mean = 3.0/(1.0/x + 1.0/y + 1.0/z)

WRITE(*,*) 'Arithmetic mean = ', arith_mean
WRITE(*,*) 'Geometric mean  = ', geo_mean
WRITE(*,*) 'Harmonic mean   = ', harm_mean

END PROGRAM ComputeMeans
