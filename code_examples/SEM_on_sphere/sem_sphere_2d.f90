SUBROUTINE update_element(n, ngq, dvvt_flat, jac, Ainv, vel, psi, ret_psi)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, ngq
  REAL(8), INTENT(IN) :: dvvt_flat(ngq*ngq), jac(n), Ainv(n*4), vel(n*2), psi(n)
  REAL(8), INTENT(INOUT) :: ret_psi(n)

  INTEGER :: nelem, ie, idx, gi, gj, k
  REAL(8) :: vpsix(ngq,ngq), vpsiy(ngq,ngq), dvvt(ngq,ngq)
  REAL(8) :: tmpx, tmpy


  nelem = n/(ngq*ngq)
  dvvt = RESHAPE(dvvt_flat, (/ngq,ngq/))


!  DO i=0,n-1
!    ie = i/(ngq*ngq)
!    gi = MOD(i,ngq*ngq)/ngq
!    gj = MOD(i,ngq)
!  END DO


  DO ie=1,nelem
    DO gi=1,ngq
      DO gj=1,ngq
        idx = (ie-1)*ngq*ngq + (gi-1)*ngq + (gj-1) + 1

        vpsix(gi,gj) = (Ainv(idx*4+0)*vel(idx*2+0) + &
                        Ainv(idx*4+1)*vel(idx*2+1)) * psi(idx)*jac(idx)

        vpsiy(gi,gj) = (Ainv(idx*4+2)*vel(idx*2+0) + &
                        Ainv(idx*4*3)*vel(idx*2+1)) * psi(idx)*jac(idx)
      END DO
    END DO

    DO gi=1,ngq
      DO gj=1,ngq
        tmpx = 0.D0
        tmpy = 0.D0
        DO k=1,ngq
          tmpx = tmpx + dvvt(k,gi)*vpsix(k,gj)
          tmpy = tmpy + vpsiy(gi,k)*dvvt(k,gj)
        END DO

        idx = (ie-1)*ngq*ngq + (gi-1)*ngq + (gj-1) + 1
        ret_psi(idx) = -(tmpx + tmpy) / jac(idx)

      END DO
    END DO
  END DO
END SUBROUTINE
