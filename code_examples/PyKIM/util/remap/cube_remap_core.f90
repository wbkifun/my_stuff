SUBROUTINE remap_fixed_2d(mat_size, dst_size, src_size, src_address, remap_matrix, src_var, dst_var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: mat_size, dst_size, src_size
  INTEGER, INTENT(IN) :: src_address(mat_size,dst_size)
  REAL(8), INTENT(IN) :: remap_matrix(mat_size,dst_size)
  REAL(8), INTENT(IN) :: src_var(src_size)
  REAL(8), INTENT(INOUT) :: dst_var(dst_size)

  INTEGER :: dst, src, i
  REAL(8) :: wgt, lsum

  DO dst=1,dst_size
    lsum = 0.0D0
    DO i=1,mat_size
      src = src_address(i,dst) + 1
      wgt = remap_matrix(i,dst)
      lsum = lsum + src_var(src)*wgt
    END DO

    dst_var(dst) = lsum
  END DO
END SUBROUTINE




SUBROUTINE remap_fixed_3d(mat_size, dst_size, src_size, nlev, src_address, remap_matrix, src_var, dst_var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: mat_size, dst_size, src_size, nlev
  INTEGER, INTENT(IN) :: src_address(mat_size,dst_size)
  REAL(8), INTENT(IN) :: remap_matrix(mat_size,dst_size)
  REAL(8), INTENT(IN) :: src_var(src_size*nlev)
  REAL(8), INTENT(INOUT) :: dst_var(dst_size*nlev)

  INTEGER :: dst, src, lev, i, idx
  REAL(8) :: wgt, lsum

  DO lev=0,nlev-1
    DO dst=1,dst_size
      lsum = 0.0D0
      DO i=1,mat_size
        src = src_address(i,dst) + 1
        wgt = remap_matrix(i,dst)
        idx = src + lev*src_size
        lsum = lsum + src_var(idx)*wgt
      END DO

      idx = dst + lev*dst_size
      dst_var(idx) = lsum
    END DO
  END DO
END SUBROUTINE 




SUBROUTINE remap_fixed_4d(mat_size, dst_size, src_size, nlev, ntime, src_address, remap_matrix, src_var, dst_var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: mat_size, dst_size, src_size, nlev, ntime
  INTEGER, INTENT(IN) :: src_address(mat_size,dst_size)
  REAL(8), INTENT(IN) :: remap_matrix(mat_size,dst_size)
  REAL(8), INTENT(IN) :: src_var(src_size*nlev*ntime)
  REAL(8), INTENT(INOUT) :: dst_var(dst_size*nlev*ntime)

  INTEGER :: dst, src, time, lev, i, idx
  REAL(8) :: wgt, lsum

  DO time=0,ntime-1
    DO lev=0,nlev-1
      DO dst=1,dst_size
        lsum = 0.0D0
        DO i=1,mat_size
          src = src_address(i,dst) + 1
          wgt = remap_matrix(i,dst)
          idx = src + lev*src_size + time*lev*src_size
          lsum = lsum + src_var(idx)*wgt
        END DO

        idx = dst + lev*dst_size + time*lev*dst_size
        dst_var(idx) = lsum
      END DO
    END DO
  END DO
END SUBROUTINE




SUBROUTINE remap_vgecore_2d(dst_size, src_size, num_links, dst_address, src_address, remap_matrix, src_var, dst_var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: dst_size, src_size, num_links
  INTEGER, INTENT(IN) :: dst_address(num_links)
  INTEGER, INTENT(IN) :: src_address(num_links)
  REAL(8), INTENT(IN) :: remap_matrix(num_links)
  REAL(8), INTENT(IN) :: src_var(src_size)
  REAL(8), INTENT(INOUT) :: dst_var(dst_size)

  INTEGER :: dst, src, i
  REAL(8) :: wgt

  DO i=1,num_links
    dst = dst_address(i) + 1
    src = src_address(i) + 1
    wgt = remap_matrix(i)
    dst_var(dst) = dst_var(dst) + src_var(src)*wgt
  END DO
END SUBROUTINE




SUBROUTINE remap_vgecore_3d(dst_size, src_size, num_links, nlev, dst_address, src_address, remap_matrix, src_var, dst_var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: dst_size, src_size, num_links, nlev
  INTEGER, INTENT(IN) :: dst_address(num_links)
  INTEGER, INTENT(IN) :: src_address(num_links)
  REAL(8), INTENT(IN) :: remap_matrix(num_links)
  REAL(8), INTENT(IN) :: src_var(src_size*nlev)
  REAL(8), INTENT(INOUT) :: dst_var(dst_size*nlev)

  INTEGER :: dst, src, lev, i, src_idx, dst_idx
  REAL(8) :: wgt

  DO lev=0,nlev-1
    DO i=1,num_links
      dst = dst_address(i) + 1
      src = src_address(i) + 1
      wgt = remap_matrix(i)

      src_idx = src + lev*src_size
      dst_idx = dst + lev*dst_size
      dst_var(dst_idx) = dst_var(dst_idx) + src_var(src_idx)*wgt
    END DO
  END DO
END SUBROUTINE




SUBROUTINE remap_vgecore_4d(dst_size, src_size, num_links, nlev, ntime,&
    dst_address, src_address, remap_matrix, src_var, dst_var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: dst_size, src_size, num_links, nlev, ntime
  INTEGER, INTENT(IN) :: dst_address(num_links)
  INTEGER, INTENT(IN) :: src_address(num_links)
  REAL(8), INTENT(IN) :: remap_matrix(num_links)
  REAL(8), INTENT(IN) :: src_var(src_size*nlev*ntime)
  REAL(8), INTENT(INOUT) :: dst_var(dst_size*nlev*ntime)

  INTEGER :: dst, src, time, lev, i, src_idx, dst_idx
  REAL(8) :: wgt

  DO time=1,ntime
    DO lev=1,nlev
      DO i=1,num_links
        dst = dst_address(i) + 1
        src = src_address(i) + 1
        wgt = remap_matrix(i)

        src_idx = src + lev*src_size + time*lev*src_size
        dst_idx = dst + lev*dst_size + time*lev*dst_size
        dst_var(dst_idx) = dst_var(dst_idx) + src_var(src_idx)*wgt
      END DO
    END DO
  END DO
END SUBROUTINE




SUBROUTINE remap_dominant_2d(dst_size, src_size, num_links, dst_address, src_address, remap_matrix, src_var, dst_var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: dst_size, src_size, num_links
  INTEGER, INTENT(IN) :: dst_address(num_links)
  INTEGER, INTENT(IN) :: src_address(num_links)
  REAL(8), INTENT(IN) :: remap_matrix(num_links)
  INTEGER, INTENT(IN) :: src_var(src_size)
  INTEGER, INTENT(INOUT) :: dst_var(dst_size)

  INTEGER :: dst, src, prev_dst, i, type_idx
  REAL(8) :: wgt
  REAL(8) :: types(24)   ! max number of types

  types(:) = 0.0D0
  prev_dst = dst_address(1) + 1

  DO i=1,num_links
    dst = dst_address(i) + 1
    src = src_address(i) + 1
    wgt = remap_matrix(i)

    IF (dst .NE. prev_dst) THEN
      dst_var(prev_dst) = MAXLOC(types, 1)
      prev_dst = dst
      types(:) = 0.0D0
    ELSE IF (i .EQ. num_links) THEN
      dst_var(dst) = MAXLOC(types, 1)
    END IF

    type_idx = src_var(src)
    types(type_idx) = types(type_idx) + wgt
  END DO
END SUBROUTINE
