MODULE derivative
  IMPLICIT NONE

CONTAINS
  SUBROUTINE gradient(ngq, i, j, rr_earth, Dinv, Dvv, scalar, ret)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: ngq, i, j
    REAL(8), INTENT(IN) :: rr_earth
    REAL(8), INTENT(IN) :: Dinv(4)
    REAL(8), INTENT(IN) :: Dvv(ngq*ngq)
    REAL(8), INTENT(IN) :: scalar(ngq*ngq)
    REAL(8), INTENT(INOUT) :: ret(2)

    INTEGER :: k
    REAL(8) :: dsdx, dsdy

    dsdx = 0.D0
    dsdy = 0.D0
    DO k=0,ngq-1
      dsdx = dsdx + Dvv(i*ngq+k+1)*scalar(j*ngq+k+1)  !(k,i), (k,j)
      dsdy = dsdy + Dvv(j*ngq+k+1)*scalar(k*ngq+i+1)  !(k,j), (i,k)
    END DO
    dsdx = dsdx*rr_earth
    dsdy = dsdy*rr_earth

    ! co-variant -> latlon (DIT)
    ret(1) = Dinv(1)*dsdx + Dinv(3)*dsdy  ! lon
    ret(2) = Dinv(2)*dsdx + Dinv(4)*dsdy  ! lat
  END SUBROUTINE gradient


  FUNCTION divergence(ngq, i, j, rr_earth, jac, Dinv, Dvv, v1, v2) result(ret)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: ngq, i, j
    REAL(8), INTENT(IN) :: rr_earth, jac
    REAL(8), INTENT(IN) :: Dinv(4)
    REAL(8), INTENT(IN) :: Dvv(ngq*ngq)
    REAL(8), INTENT(IN) :: v1(ngq*ngq), v2(ngq*ngq)
    REAL(8) :: ret

    INTEGER :: k
    REAL(8) :: gv1(ngq*ngq), gv2(ngq*ngq)
    REAL(8) :: d1dx, d2dy

    ! latlon -> contra-variant (DI)
    gv1 = jac*(Dinv(1)*v1 + Dinv(2)*v2)
    gv2 = jac*(Dinv(3)*v1 + Dinv(4)*v2)

    d1dx = 0.D0
    d2dy = 0.D0
    DO k=0,ngq-1
      !d1dx = d1dx + Dvv(i*ngq+k+1)*gv1(j*ngq+k+1)  !(k,i), (k,j)
      d1dx = d1dx + v1(j*ngq+0+1)  !(k,i), (k,j)
      d2dy = d2dy + Dvv(j*ngq+k+1)*gv2(k*ngq+i+1)  !(k,j), (i,k)
    END DO

    !ret = (d1dx + d2dy)*rr_earth/jac
    ret = v1(j*ngq+0+1)
  END FUNCTION divergence


  FUNCTION vorticity(ngq, i, j, rr_earth, jac, Dinv, Dvv, v1, v2) result(ret)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: ngq, i, j
    REAL(8), INTENT(IN) :: rr_earth, jac
    REAL(8), INTENT(IN) :: Dinv(4)
    REAL(8), INTENT(IN) :: Dvv(ngq*ngq)
    REAL(8), INTENT(IN) :: v1(ngq*ngq), v2(ngq*ngq)
    REAL(8) :: ret

    INTEGER :: k
    REAL(8) :: a, b, c, d, rdet
    REAL(8) :: DT(4)
    REAL(8) :: gv1(ngq*ngq), gv2(ngq*ngq)
    REAL(8) :: d1dx, d2dy

    ! D from Dinv
    a = Dinv(1)
    b = Dinv(2)
    c = Dinv(3)
    d = Dinv(4)
    rdet = 1.D0/(a*d-b*c)
    DT(1) =   rdet*d
    DT(2) = - rdet*c  ! transpose
    DT(3) = - rdet*b  ! transpose
    DT(4) =   rdet*a

    ! latlon -> co-variant (DT)
    gv1 = DT(1)*v1 + DT(2)*v2
    gv2 = DT(3)*v1 + DT(4)*v2

    d1dx = 0.D0
    d2dy = 0.D0
    DO k=0,ngq-1
      d1dx = d1dx + Dvv(i*ngq+k+1)*gv1(j*ngq+k+1)  !(k,i), (k,j)
      d2dy = d2dy + Dvv(j*ngq+k+1)*gv2(k*ngq+i+1)  !(k,j), (i,k)
    END DO

    ret = (d1dx - d2dy)*rr_earth/jac
  END FUNCTION vorticity
END MODULE




SUBROUTINE compute_rhs(nsize, nlev, ngq, rr_earth, ps0, &
    Dvv, jac, Dinv, hyai, hybi, hyam, hybm, etai, &
    ps_v, grad_ps, v1, v2, divdp, vort, tmp3d)
  USE derivative, ONLY : gradient, divergence, vorticity
  
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nsize, nlev, ngq  !number of horizontal / vertical / Gauss-quadrature points
  REAL(8), INTENT(IN) :: rr_earth, ps0
  REAL(8), INTENT(IN) :: Dvv(ngq*ngq), jac(nsize), Dinv(nsize*4)
  REAL(8), INTENT(IN) :: hyai(nlev+1), hybi(nlev+1), hyam(nlev), hybm(nlev), etai(nlev+1)
  REAL(8), INTENT(IN) :: ps_v(nsize)
  REAL(8), INTENT(IN) :: v1(nsize*nlev), v2(nsize*nlev)
  REAL(8), INTENT(INOUT) :: grad_ps(nsize*2)
  REAL(8), INTENT(INOUT) :: divdp(nsize*nlev), vort(nsize*nlev)
  REAL(8), INTENT(INOUT) :: tmp3d(nsize*nlev)


  INTEGER :: ngq2, idx, gi, gj, ielem, vidx, vk
  INTEGER :: ij, i, j, k, l
  REAL(8) :: jac_pt, Dinv_pt(4)
  REAL(8) :: elem_ps_v(ngq*ngq)
  REAL(8) :: elem_v1(ngq*ngq), elem_v2(ngq*ngq)
  REAL(8) :: elem_dp(ngq*ngq), elem_v1dp(ngq*ngq), elem_v2dp(ngq*ngq)
  REAL(8) :: ret_vec(2)
  REAL(8) :: dn, p, grad_p1, grad_p2, dpdn, rdp, rdpdn
  REAL(8) :: vgrad_p


  ngq2 = ngq*ngq

  DO idx=0,nsize-1
    gi = MOD(idx,ngq)        ! inmost index in an element
    gj = MOD(idx,ngq2)/ngq
    ielem = idx/ngq2
    ij = gj*ngq + gi + 1      ! index in an element (2D)

    Dinv_pt = Dinv(idx*4+1:idx*4+4)
    jac_pt = jac(idx+1)

    IF (gi==0 .AND. gj==0) THEN
      elem_ps_v = ps_v(idx+1:idx+ngq2)
    END IF


    !--------------------------------------------------------------------------
    ! Compute pressure (p) on half levels from ps 
    ! using the hybrid coordinates relationship
    !--------------------------------------------------------------------------
    CALL gradient(ngq, gi, gj, rr_earth, Dinv_pt, Dvv, elem_ps_v, ret_vec)
    grad_ps(idx*2+1) = ret_vec(1)
    grad_ps(idx*2+2) = ret_vec(2)


    !--------------------------------------------
    ! Compute p and delta p
    !--------------------------------------------
    DO vk=0,nlev-1
      vidx = ielem*ngq2*nlev + vk*ngq2 + gj*ngq + gi
      k = vk + 1

      IF (gi==0 .AND. gj==0) THEN
        elem_v1 = v1(vidx+1:vidx+ngq2)
        elem_v2 = v2(vidx+1:vidx+ngq2)

        elem_dp = &
          (hyai(k+1)*ps0 + hybi(k+1)*elem_ps_v) - &
          (hyai(k  )*ps0 + hybi(k  )*elem_ps_v) 

        elem_v1dp = elem_v1*elem_dp
        elem_v2dp = elem_v2*elem_dp

        tmp3d(vidx+1:vidx+ngq2) = elem_v1dp
      END IF

      dn = etai(k+1) - etai(k)
      p = hyam(k)*ps0 + hybm(k)*elem_ps_v(ij)
      grad_p1 = hybm(k)*grad_ps(idx*2+1)
      grad_p2 = hybm(k)*grad_ps(idx*2+2)
      !dpdn = dp/dn
      !rdp = 1.D0/dp
      !rdpdn = 1.D0/dpdn

      !----------------------------------
      ! Compute vgrad_lnps
      !----------------------------------
      vgrad_p = elem_v1(ij)*grad_p1 + elem_v2(ij)*grad_p2

      !----------------------------------
      ! Compute relative vorticity and divergence
      !----------------------------------
      divdp(vidx+1) = divergence(ngq, gi, gj, rr_earth, jac_pt, Dinv_pt, Dvv, elem_v1dp, elem_v2dp)
      vort(vidx+1) = vorticity(ngq, gi, gj, rr_earth, jac_pt, Dinv_pt, Dvv, elem_v1, elem_v2)
    END DO
  END DO
END SUBROUTINE compute_rhs
