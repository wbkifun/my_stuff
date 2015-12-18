PROGRAM indices
  IMPLICIT NONE

  INTEGER :: n, ngq, ngq2, nlev
  INTEGER :: idx, i, j, k, ielem, ipilr

  ngq = 4
  nlev = 3
  ngq2 = ngq*ngq
  n = 17*ngq*ngq

  PRINT *, 'idx, i, j, k, ielem, ipilr'

  DO idx=0,n
    i = MOD(idx,ngq)
    j = MOD(idx,ngq2)/ngq
    k = MOD(idx,ngq2*nlev)/ngq2

    ielem = idx/(ngq2)
    ipilr = idx/(ngq2*nlev)

    PRINT *, idx, i, j, k, ielem, ipilr
  END DO
END PROGRAM
