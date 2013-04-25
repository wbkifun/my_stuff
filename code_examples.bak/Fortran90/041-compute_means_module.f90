MODULE ComputeMeansModule
IMPLICIT NONE

CONTAINS

REAL FUNCTION ArithMean(x, y, z)
IMPLICIT NONE
REAL, INTENT(IN) :: x, y, z

ArithMean = (x + y + z)/3.0
END FUNCTION ArithMean

REAL FUNCTION GeoMean(x, y, z)
IMPLICIT NONE
REAL, INTENT(IN) :: x, y, z

GeoMean = (x * y * z)**(1.0/3.0)
END FUNCTION GeoMean

REAL FUNCTION HarmMean(x, y, z)
IMPLICIT NONE
REAL, INTENT(IN) :: x, y, z

HarmMean = 3.0/(1.0/x + 1.0/y + 1.0/z)
END FUNCTION HarmMean

END MODULE ComputeMeansModule
