SUBROUTINE copy_grid2buf_f90(nelem, ngll, nlev, size_table, size_buf, num_var, var_idx, grid2buf, buf, var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, nlev, size_table, size_buf, num_var, var_idx
  INTEGER, DIMENSION(3,size_table), INTENT(IN) :: grid2buf
  DOUBLE PRECISION, DIMENSION(nlev, size_buf, num_var), INTENT(INOUT) :: buf
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(INOUT) :: var

  INTEGER :: i, lev
  INTEGER :: gi, gj, elem
  

  DO i=1,size_table
    gi = grid2buf(1,i)
    gj = grid2buf(2,i)
    elem = grid2buf(3,i)

    DO lev=1,nlev
      buf(lev,i,var_idx) = var(gi,gj,lev,elem)
    END DO
  END DO
END SUBROUTINE




SUBROUTINE interact_buf_avg_f90(nelem, ngll, nlev, size_table, size_buf, num_var, var_idx, buf2grid, mvp_buf, buf, var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, nlev, size_table, size_buf, num_var, var_idx
  INTEGER, DIMENSION(3,size_table), INTENT(IN) :: buf2grid
  INTEGER, DIMENSION(4,size_table), INTENT(IN) :: mvp_buf
  DOUBLE PRECISION, DIMENSION(nlev, size_buf, num_var), INTENT(IN) :: buf
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(INOUT) :: var

  INTEGER :: i, lev
  INTEGER :: gi, gj, elem
  INTEGER :: bi1, bi2, bi3, bi4
  

  DO i=1,size_table
    gi = buf2grid(1,i)
    gj = buf2grid(2,i)
    elem = buf2grid(3,i)

    bi1 = mvp_buf(1,i)
    bi2 = mvp_buf(2,i)
    bi3 = mvp_buf(3,i)
    bi4 = mvp_buf(4,i)

    DO lev=1,nlev
      IF( mvp_buf(3,i) == -1 ) THEN
        var(gi,gj,lev,elem) = (buf(lev,bi1,var_idx) + buf(lev,bi2,var_idx))/2.D0

      ELSE IF( mvp_buf(4,i) == -1 ) THEN
        var(gi,gj,lev,elem) = (buf(lev,bi1,var_idx) + buf(lev,bi2,var_idx) + &
                               buf(lev,bi3,var_idx))/3.D0

      ELSE
        var(gi,gj,lev,elem) = (buf(lev,bi1,var_idx) + buf(lev,bi2,var_idx) + &
                               buf(lev,bi3,var_idx) + buf(lev,bi4,var_idx))/4.D0
      END IF
    END DO
  END DO
END SUBROUTINE




SUBROUTINE interact_inner_avg_f90(nelem, ngll, nlev, size_table, mvp_inner, var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, nlev, size_table
  INTEGER, DIMENSION(3,4,size_table), INTENT(IN) :: mvp_inner
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(INOUT) :: var

  DOUBLE PRECISION :: sv
  INTEGER :: i, lev
  INTEGER :: gi1, gi2, gi3, gi4
  INTEGER :: gj1, gj2, gj3, gj4
  INTEGER :: elem1, elem2, elem3, elem4
  

  DO i=1,size_table
    gi1 = mvp_inner(1,1,i)
    gj1 = mvp_inner(2,1,i)
    elem1 = mvp_inner(3,1,i)

    gi2 = mvp_inner(1,2,i)
    gj2 = mvp_inner(2,2,i)
    elem2 = mvp_inner(3,2,i)

    gi3 = mvp_inner(1,3,i)
    gj3 = mvp_inner(2,3,i)
    elem3 = mvp_inner(3,3,i)

    gi4 = mvp_inner(1,4,i)
    gj4 = mvp_inner(2,4,i)
    elem4 = mvp_inner(3,4,i)

    DO lev=1,nlev
      IF( gi3 == -1 ) THEN
        sv = (var(gi1,gj1,lev,elem1) + var(gi2,gj2,lev,elem2))/2.D0
        var(gi1,gj1,lev,elem1) = sv
        var(gi2,gj2,lev,elem2) = sv

      ELSE IF( gi4 == -1 ) THEN
        sv = (var(gi1,gj1,lev,elem1) + var(gi2,gj2,lev,elem2) + &
              var(gi3,gj3,lev,elem3))/3.D0
        var(gi1,gj1,lev,elem1) = sv
        var(gi2,gj2,lev,elem2) = sv
        var(gi3,gj3,lev,elem3) = sv

      ELSE
        sv = (var(gi1,gj1,lev,elem1) + var(gi2,gj2,lev,elem2) + &
              var(gi3,gj3,lev,elem3) + var(gi4,gj4,lev,elem4))/4.D0
        var(gi1,gj1,lev,elem1) = sv
        var(gi2,gj2,lev,elem2) = sv
        var(gi3,gj3,lev,elem3) = sv
        var(gi4,gj4,lev,elem4) = sv
      END IF
    END DO
  END DO
END SUBROUTINE
