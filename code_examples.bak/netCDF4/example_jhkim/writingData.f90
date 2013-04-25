PROGRAM test
USE NETCDF
IMPLICIT NONE
!INCLUDE 'netcdf.inc'
CHARACTER (len=20)   :: filename
INTEGER, PARAMETER   :: n = 20
INTEGER              :: u(n)
INTEGER              :: dimID, varID
INTEGER              :: i, cmode, ncid, ncstat


DO i = 1, n
   u(i) = 2*i
ENDDO


! Create the File
filename = './Data.nc'
cmode = NF90_CLOBBER
ncstat = NF90_CREATE(filename, cmode, ncid)
IF(ncstat == NF90_NOERR) THEN
   WRITE(*,*) 'File Creation was Completed ...'
END IF

ncstat = NF90_DEF_DIM(ncid, "dimSize", n, dimID)
ncstat = NF90_DEF_VAR(ncid, "variable", NF90_INT, (/dimID/), varID)
ncstat = NF90_PUT_ATT(ncid, varID, "unit", "m/s")

! Open NetCDF Dataset
ncstat = NF90_ENDDEF(ncid)

ncstat = NF90_PUT_VAR(ncid, varID, u)

ncstat = NF90_CLOSE(ncid)
END PROGRAM test
