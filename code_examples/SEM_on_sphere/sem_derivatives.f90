SUBROUTINE gradient(n, ngq, dvvT, jac, Ainv, psi, ret)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, ngq
  REAL(8), INTENT(IN) :: dvvT(ngq*ngq), jac(n), Ainv(n*4), psi(n)
  REAL(8), INTENT(INOUT) :: ret(n*2)

  INTEGER :: idx, i, j, k
  REAL(8) :: elem_psi(ngq*ngq)
  REAL(8) :: tmpx, tmpy


  DO idx=0,n-1
    i = MOD(idx,ngq)          ! inmost index
    j = MOD(idx,ngq*ngq)/ngq  ! outmost index

    IF (i==0 .AND. j==0) THEN
      DO k=1,ngq*ngq
        elem_psi(k) = psi(idx+k)
      END DO
    END IF

    tmpx = 0.D0
    tmpy = 0.D0
    DO k=0,ngq-1
      tmpx = tmpx + dvvT(i*ngq+k+1)*elem_psi(j*ngq+k+1)
      tmpy = tmpy + elem_psi(k*ngq+i+1)*dvvT(j*ngq+k+1)
    END DO

    ! covariant -> latlon (AIT)
    ret(idx*2+1) = Ainv(idx*4+1)*tmpx + Ainv(idx*4+3)*tmpy  ! lon
    ret(idx*2+2) = Ainv(idx*4+2)*tmpx + Ainv(idx*4+4)*tmpy  ! lat
  END DO
END SUBROUTINE
