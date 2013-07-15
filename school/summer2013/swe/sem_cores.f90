SUBROUTINE compute_rhs_f90(nelem, ngll, nlev, rrearth, gconst, dvvT, jac, A, Ainv, fcor, u, v, h, ret_u, ret_v, ret_h)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nelem, ngll, nlev
  DOUBLE PRECISION, INTENT(IN) :: rrearth, gconst
  DOUBLE PRECISION, DIMENSION(ngll,ngll), INTENT(IN) :: dvvT
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nelem), INTENT(IN) :: jac
  DOUBLE PRECISION, DIMENSION(2,2,ngll,ngll,nelem), INTENT(IN) :: A, Ainv
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nelem), INTENT(IN) :: fcor
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(IN) :: u, v, h
  DOUBLE PRECISION, DIMENSION(ngll,ngll,nlev,nelem), INTENT(INOUT) :: ret_u, ret_v, ret_h

  INTEGER :: ie, gi, gj, k
  INTEGER :: lev
  DOUBLE PRECISION :: tmpx, tmpy
  DOUBLE PRECISION, DIMENSION(ngll,ngll) :: uu, vv, hh, jj
  DOUBLE PRECISION, DIMENSION(ngll,ngll) :: E, vx_co, vy_co, jhvx_contra, jhvy_contra
  DOUBLE PRECISION, DIMENSION(ngll,ngll) :: grad_lon, grad_lat, vort, div
  

  lev = 1

  DO ie=1,nelem
    !--------------------------------------------------------------------------
    ! common values
    !--------------------------------------------------------------------------
    DO gj=1,ngll
      DO gi=1,ngll
        uu(gi,gj) = u(gi,gj,lev,ie)
        vv(gi,gj) = v(gi,gj,lev,ie)
        hh(gi,gj) = h(gi,gj,lev,ie)
        jj(gi,gj) = jac(gi,gj,ie)
      END DO
    END DO


    !--------------------------------------------------------------------------
    ! gradient
    !--------------------------------------------------------------------------
    DO gj=1,ngll
      DO gi=1,ngll
        E(gi,gj) = 0.5D0*(uu(gi,gj)**2 + vv(gi,gj)**2) + gconst*hh(gi,gj)
      END DO
    END DO

    DO gj=1,ngll
      DO gi=1,ngll
        tmpx = 0.D0
        tmpy = 0.D0

        DO k=1,ngll
          tmpx = tmpx + dvvT(k,gi)*E(k,gj)
          tmpy = tmpy + E(gi,k)*dvvT(k,gj)
        END DO

        ! co -> latlon (AinvT)
        grad_lon(gi,gj) = rrearth*( Ainv(1,1,gi,gj,ie)*tmpx + Ainv(2,1,gi,gj,ie)*tmpy )
        grad_lat(gi,gj) = rrearth*( Ainv(1,2,gi,gj,ie)*tmpx + Ainv(2,2,gi,gj,ie)*tmpy )
      END DO
    END DO


    !--------------------------------------------------------------------------
    ! vorticity
    !--------------------------------------------------------------------------
    DO gj=1,ngll
      DO gi=1,ngll
        ! latlon -> co (AT)
        vx_co(gi,gj) = A(1,1,gi,gj,ie)*uu(gi,gj) + A(2,1,gi,gj,ie)*vv(gi,gj)
        vy_co(gi,gj) = A(1,2,gi,gj,ie)*uu(gi,gj) + A(2,2,gi,gj,ie)*vv(gi,gj)
      END DO
    END DO

    DO gj=1,ngll
      DO gi=1,ngll
        tmpx = 0.D0
        tmpy = 0.D0

        DO k=1,ngll
          tmpx = tmpx + dvvT(k,gi)*vy_co(k,gj)
          tmpy = tmpy + vx_co(gi,k)*dvvT(k,gj)
        END DO

        vort(gi,gj) = rrearth*(tmpx - tmpy)/jj(gi,gj)
      END DO
    END DO


    !--------------------------------------------------------------------------
    ! divergence
    !--------------------------------------------------------------------------
    DO gj=1,ngll
      DO gi=1,ngll
        ! latlon -> contra (Ainv)
        jhvx_contra(gi,gj) = (Ainv(1,1,gi,gj,ie)*uu(gi,gj) + Ainv(1,2,gi,gj,ie)*vv(gi,gj))*jj(gi,gj)*hh(gi,gj)
        jhvy_contra(gi,gj) = (Ainv(2,1,gi,gj,ie)*uu(gi,gj) + Ainv(2,2,gi,gj,ie)*vv(gi,gj))*jj(gi,gj)*hh(gi,gj)
      END DO
    END DO

    DO gj=1,ngll
      DO gi=1,ngll
        tmpx = 0.D0
        tmpy = 0.D0

        DO k=1,ngll
          tmpx = tmpx + dvvT(k,gi)*jhvx_contra(k,gj)
          tmpy = tmpy + jhvy_contra(gi,k)*dvvT(k,gj)
        END DO

        div(gi,gj) = rrearth*(tmpx + tmpy)/jj(gi,gj)
      END DO
    END DO


    !--------------------------------------------------------------------------
    ! result
    !--------------------------------------------------------------------------
    DO gj=1,ngll
      DO gi=1,ngll
        ret_u(gi,gj,lev,ie) = (vort(gi,gj) + fcor(gi,gj,ie))*vv(gi,gj) - grad_lon(gi,gj)
        ret_v(gi,gj,lev,ie) = -(vort(gi,gj) + fcor(gi,gj,ie))*uu(gi,gj) - grad_lat(gi,gj)
        ret_h(gi,gj,lev,ie) = -div(gi,gj)
      END DO
    END DO

  END DO
END SUBROUTINE
