PROGRAM write_arr
USE netcdf
IMPLICIT NONE

CHARACTER (len=20) :: filename='data.nc'
INTEGER, PARAMETER :: np=4, ne=10
!DOUBLE PRECISION :: v0(np)
INTEGER :: v0(np)
INTEGER :: i
INTEGER :: ncid, ncstat, dim_np, dim_ne, var_v0

DO i=1,np
  v0(i) = i
END DO

! create a file
ncstat = NF90_CREATE(filename, NF90_CLOBBER, ncid)
IF(ncstat == NF90_NOERR) THEN
    WRITE(*,*) 'File creation was completed.'
END IF

ncstat = NF90_DEF_DIM(ncid, 'np', np, dim_np)
ncstat = NF90_DEF_VAR(ncid, 'v0', NF90_INT, (/dim_np/), var_v0)
ncstat = NF90_ENDDEF(ncid)

ncstat = NF90_PUT_VAR(ncid, var_v0, v0)

ncstat = NF90_CLOSE(ncid)

END PROGRAM write_arr
