SUBROUTINE interact_between_elems_inner_f90(nelem, ngll, nlev, size2, size3, size4, mvp_inner2, mvp_inner3, mvp_inner4, var)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, nlev, size2, size3, size4
  INTEGER, DIMENSION(3,2,size2), INTENT(IN) :: mvp_inner2
  INTEGER, DIMENSION(3,3,size3), INTENT(IN) :: mvp_inner3
  INTEGER, DIMENSION(3,4,size4), INTENT(IN) :: mvp_inner4
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(INOUT) :: var

  DOUBLE PRECISION :: sv
  INTEGER :: i, lev
  INTEGER :: gi1, gi2, gi3, gi4
  INTEGER :: gj1, gj2, gj3, gj4
  INTEGER :: elem1, elem2, elem3, elem4
  

  lev = 1

  DO i=1,size2
    gi1 = mvp_inner2(1,1,i)
    gj1 = mvp_inner2(2,1,i)
    elem1 = mvp_inner2(3,1,i)

    gi2 = mvp_inner2(1,2,i)
    gj2 = mvp_inner2(2,2,i)
    elem2 = mvp_inner2(3,2,i)

    sv = (var(gi1,gj1,lev,elem1) + var(gi2,gj2,lev,elem2))/2.D0
    var(gi1,gj1,lev,elem1) = sv
    var(gi2,gj2,lev,elem2) = sv
  END DO


  DO i=1,size3
    gi1 = mvp_inner3(1,1,i)
    gj1 = mvp_inner3(2,1,i)
    elem1 = mvp_inner3(3,1,i)

    gi2 = mvp_inner3(1,2,i)
    gj2 = mvp_inner3(2,2,i)
    elem2 = mvp_inner3(3,2,i)

    gi3 = mvp_inner3(1,3,i)
    gj3 = mvp_inner3(2,3,i)
    elem3 = mvp_inner3(3,3,i)

    sv = (var(gi1,gj1,lev,elem1) + var(gi2,gj2,lev,elem2) + &
          var(gi3,gj3,lev,elem3))/3.D0
    var(gi1,gj1,lev,elem1) = sv
    var(gi2,gj2,lev,elem2) = sv
    var(gi3,gj3,lev,elem3) = sv
  END DO


  DO i=1,size4
    gi1 = mvp_inner4(1,1,i)
    gj1 = mvp_inner4(2,1,i)
    elem1 = mvp_inner4(3,1,i)

    gi2 = mvp_inner4(1,2,i)
    gj2 = mvp_inner4(2,2,i)
    elem2 = mvp_inner4(3,2,i)

    gi3 = mvp_inner4(1,3,i)
    gj3 = mvp_inner4(2,3,i)
    elem3 = mvp_inner4(3,3,i)

    gi4 = mvp_inner4(1,4,i)
    gj4 = mvp_inner4(2,4,i)
    elem4 = mvp_inner4(3,4,i)

    sv = (var(gi1,gj1,lev,elem1) + var(gi2,gj2,lev,elem2) + &
          var(gi3,gj3,lev,elem3) + var(gi4,gj4,lev,elem4))/4.D0
    var(gi1,gj1,lev,elem1) = sv
    var(gi2,gj2,lev,elem2) = sv
    var(gi3,gj3,lev,elem3) = sv
    var(gi4,gj4,lev,elem4) = sv
  END DO
END SUBROUTINE
