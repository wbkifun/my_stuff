PROGRAM write_arr
USE netcdf
IMPLICIT NONE

CHARACTER (len=20) :: filename='nc_test_01.nc'
INTEGER, PARAMETER :: np=4, ne=10
INTEGER :: v0(np), v1(np,np), v2(np,np,ne)
INTEGER :: dim_id, var_id
INTEGER :: i, j, ie 
INTEGER :: nc_id, ncstat, dim_np, dim_ne, var_v0, var_v1, var_v2

DO i=1,np
  v0(i) = i
END DO

DO j=1,np
  DO i=1,np
    v1(i,j) = i + 100*j
  END DO
END DO

DO ie=1,ne
  DO j=1,np
    DO i=1,np
      v2(i,j,ie) = i + 100*j + 10000*ie
    END DO
  END DO
END DO

! create a file
ncstat = NF90_CREATE(filename, NF90_CLOBBER, nc_id)
IF(ncstat == NF90_NOERR) THEN
    WRITE(*,*) 'File creation was completed.'
END IF

ncstat = NF90_DEF_DIM(nc_id, 'np', np, dim_np)
ncstat = NF90_DEF_DIM(nc_id, 'ne', ne, dim_ne)

ncstat = NF90_DEF_VAR(nc_id, 'v0', NF90_INT, (/dim_np/), var_v0)
ncstat = NF90_DEF_VAR(nc_id, 'v1', NF90_INT, (/dim_np,dim_np/), var_v1)
ncstat = NF90_DEF_VAR(nc_id, 'v2', NF90_INT, (/dim_np,dim_np,dim_ne/), var_v2)

ncstat = NF90_ENDDEF(nc_id)

ncstat = NF90_PUT_VAR(nc_id, var_v0, v0)
ncstat = NF90_PUT_VAR(nc_id, var_v1, v1)
ncstat = NF90_PUT_VAR(nc_id, var_v2, v2)

ncstat = NF90_CLOSE(nc_id)

END PROGRAM write_arr
