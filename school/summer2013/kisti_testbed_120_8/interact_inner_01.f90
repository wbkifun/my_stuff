SUBROUTINE interact_between_elems_inner_f90(nelem, ngll, nlev, size_table, mvp_inner, var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, nlev, size_table
  INTEGER, DIMENSION(3,4,size_table), INTENT(IN) :: mvp_inner
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(INOUT) :: var

  DOUBLE PRECISION :: sv
  INTEGER :: i, lev
  INTEGER :: gi1, gi2, gi3, gi4
  INTEGER :: gj1, gj2, gj3, gj4
  INTEGER :: elem1, elem2, elem3, elem4
  

  lev = 1

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

  IF( gi3 == 0 ) THEN
    sv = (var(gi1,gj1,lev,elem1) + var(gi2,gj2,lev,elem2))/2.D0
    var(gi1,gj1,lev,elem1) = sv
    var(gi2,gj2,lev,elem2) = sv

  ELSE IF( gi4 == 0 ) THEN
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
END SUBROUTINE
