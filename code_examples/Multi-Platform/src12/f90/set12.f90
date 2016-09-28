
SUBROUTINE calc_divv(np, nlev, nelem, ru)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: np, nlev, nelem
  REAL(8), DIMENSION(np,np,nlev+1,nelem), INTENT(INOUT) :: ru

  INTEGER :: ie, k

  DO ie=1,nelem
    DO k=1,nlev+1
      ru(:,:,k,ie) = 1.2D0
    END DO
  END DO
END SUBROUTINE
    
