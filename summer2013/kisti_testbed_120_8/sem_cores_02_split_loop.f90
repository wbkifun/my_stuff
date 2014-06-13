SUBROUTINE compute_rhs_f90(nelem, ngll, nlev, dvvt, jac, Ainv, vel, psi, ret_psi)
  USE OMP_LIB
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, nlev
  DOUBLE PRECISION, DIMENSION(ngll,ngll), INTENT(IN) :: dvvt
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nelem), INTENT(IN) :: jac
  DOUBLE PRECISION, DIMENSION(2,2,ngll,ngll,nelem), INTENT(IN) :: Ainv
  DOUBLE PRECISION, DIMENSION(2,ngll,ngll,nlev,nelem), INTENT(IN) :: vel
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(IN) :: psi
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(INOUT) :: ret_psi

  INTEGER :: ie, gi, gj, k
  INTEGER :: lev
  DOUBLE PRECISION :: tmpx, tmpy
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nelem) :: vpsix, vpsiy
  

  lev = 1

  DO ie=1,nelem
    DO gj=1,ngll
      DO gi=1,ngll
        vpsix(gi,gj,ie) = (Ainv(1,1,gi,gj,ie)*vel(1,gi,gj,lev,ie) + &
                        Ainv(1,2,gi,gj,ie)*vel(2,gi,gj,lev,ie)) * &
                        psi(gi,gj,lev,ie)*jac(gi,gj,ie)

        vpsiy(gi,gj,ie) = (Ainv(2,1,gi,gj,ie)*vel(1,gi,gj,lev,ie) + &
                        Ainv(2,2,gi,gj,ie)*vel(2,gi,gj,lev,ie)) * &
                        psi(gi,gj,lev,ie)*jac(gi,gj,ie)
      END DO
    END DO
  END DO

  DO ie=1,nelem
    DO gj=1,ngll
      DO gi=1,ngll
        tmpx = 0.D0
        tmpy = 0.D0

        DO k=1,ngll
          tmpx = tmpx + dvvt(k,gi)*vpsix(k,gj,ie)
          tmpy = tmpy + vpsiy(gi,k,ie)*dvvt(k,gj)
        END DO

        ret_psi(gi,gj,lev,ie) = -(tmpx + tmpy) / jac(gi,gj,ie)

      END DO
    END DO
  END DO
END SUBROUTINE



SUBROUTINE add(nelem, ngll, nlev, coeff, k, psi, ret_psi)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, nlev
  DOUBLE PRECISION, INTENT(IN) :: coeff
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(IN) :: k, psi
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(INOUT) :: ret_psi

  INTEGER :: ie, gi, gj
  INTEGER :: lev

  lev = 1

  DO ie=1,nelem
    DO gj=1,ngll
      DO gi=1,ngll
        ret_psi(gi,gj,lev,ie) = coeff * k(gi,gj,lev,ie) + psi(gi,gj,lev,ie)
      END DO
    END DO
  END DO
END SUBROUTINE



SUBROUTINE rk4_add(nelem, ngll, nlev, dt, k1, k2, k3, k4, psi)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, nlev
  DOUBLE PRECISION, INTENT(IN) :: dt
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(IN) :: k1, k2, k3, k4
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(INOUT) :: psi

  INTEGER :: ie, gi, gj
  INTEGER :: lev
  DOUBLE PRECISION:: coeff

  lev = 1
  coeff = dt/6

  DO ie=1,nelem
    DO gj=1,ngll
      DO gi=1,ngll
        psi(gi,gj,lev,ie) = psi(gi,gj,lev,ie) + &
                         coeff * (k1(gi,gj,lev,ie) + 2*k2(gi,gj,lev,ie) + &
                                  2*k3(gi,gj,lev,ie) + k4(gi,gj,lev,ie))
      END DO
    END DO
  END DO
END SUBROUTINE
