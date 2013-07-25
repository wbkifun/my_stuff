!==========================================================================
! without vertical level
!==========================================================================

SUBROUTINE interact_buf_avg_f90(nelem, ngll, size_table, size_buf, num_var, var_idx, buf2grid, mvp_buf, buf, var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, size_table, size_buf, num_var, var_idx
  INTEGER, DIMENSION(3,size_table), INTENT(IN) :: buf2grid
  INTEGER, DIMENSION(4,size_table), INTENT(IN) :: mvp_buf
  DOUBLE PRECISION, DIMENSION(size_buf, num_var), INTENT(IN) :: buf
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nelem), INTENT(INOUT) :: var

  INTEGER :: i
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

    IF( bi3 == -1 ) THEN
      var(gi,gj,elem) = (buf(bi1,var_idx) + buf(bi2,var_idx))/2.D0

    ELSE IF( bi4 == -1 ) THEN
      var(gi,gj,elem) = (buf(bi1,var_idx) + buf(bi2,var_idx) + &
                         buf(bi3,var_idx))/3.D0

    ELSE
      var(gi,gj,elem) = (buf(bi1,var_idx) + buf(bi2,var_idx) + &
                         buf(bi3,var_idx) + buf(bi4,var_idx))/4.D0
    END IF
  END DO
END SUBROUTINE




SUBROUTINE interact_inner2_avg_f90(nelem, ngll, size_table, mvp_inner, var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, size_table
  INTEGER, DIMENSION(3,2,size_table), INTENT(IN) :: mvp_inner
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nelem), INTENT(INOUT) :: var

  DOUBLE PRECISION :: sv
  INTEGER :: i
  INTEGER :: gi1, gi2
  INTEGER :: gj1, gj2
  INTEGER :: elem1, elem2
  

  DO i=1,size_table
    gi1 = mvp_inner(1,1,i)
    gj1 = mvp_inner(2,1,i)
    elem1 = mvp_inner(3,1,i)

    gi2 = mvp_inner(1,2,i)
    gj2 = mvp_inner(2,2,i)
    elem2 = mvp_inner(3,2,i)

    sv = (var(gi1,gj1,elem1) + var(gi2,gj2,elem2))/2.D0
    var(gi1,gj1,elem1) = sv
    var(gi2,gj2,elem2) = sv
  END DO
END SUBROUTINE




SUBROUTINE interact_inner3_avg_f90(nelem, ngll, size_table, mvp_inner, var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, size_table
  INTEGER, DIMENSION(3,3,size_table), INTENT(IN) :: mvp_inner
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nelem), INTENT(INOUT) :: var

  DOUBLE PRECISION :: sv
  INTEGER :: i
  INTEGER :: gi1, gi2, gi3
  INTEGER :: gj1, gj2, gj3
  INTEGER :: elem1, elem2, elem3
  

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

    sv = (var(gi1,gj1,elem1) + var(gi2,gj2,elem2) + &
          var(gi3,gj3,elem3))/3.D0
    var(gi1,gj1,elem1) = sv
    var(gi2,gj2,elem2) = sv
    var(gi3,gj3,elem3) = sv
  END DO
END SUBROUTINE




SUBROUTINE interact_inner4_avg_f90(nelem, ngll, size_table, mvp_inner, var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, size_table
  INTEGER, DIMENSION(3,4,size_table), INTENT(IN) :: mvp_inner
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nelem), INTENT(INOUT) :: var

  DOUBLE PRECISION :: sv
  INTEGER :: i
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

    sv = (var(gi1,gj1,elem1) + var(gi2,gj2,elem2) + &
          var(gi3,gj3,elem3) + var(gi4,gj4,elem4))/4.D0
    var(gi1,gj1,elem1) = sv
    var(gi2,gj2,elem2) = sv
    var(gi3,gj3,elem3) = sv
    var(gi4,gj4,elem4) = sv
  END DO
END SUBROUTINE
