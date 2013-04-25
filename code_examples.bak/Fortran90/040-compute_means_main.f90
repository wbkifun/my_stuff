PROGRAM ComputeMeans
USE ComputeMeansModule
IMPLICIT NONE

REAL :: x, y, z

READ(*,*) x, y, z
WRITE(*,*) 'Input: ', x, y, z
WRITE(*,*) 
WRITE(*,*) 'Arithmetic mean = ', ArithMean(x,y,z)
WRITE(*,*) 'Geometric mean  = ', GeoMean(x,y,z)
WRITE(*,*) 'Harmonic mean   = ', HarmMean(x,y,z)

END PROGRAM ComputeMeans
