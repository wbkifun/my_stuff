PROGRAM writestn
  IMPLICIT NONE
  CHARACTER(LEN=8) :: stnid, id
  REAL(KIND=4) :: lat, lon, time, prcp, temp
  INTEGER :: date, nlev, nflag
  INTEGER :: it, i, j
  INTEGER :: istat1, istat2


  OPEN(21, FILE='asstn.bin', FORM='unformatted', STATUS='unknown', ACCESS='stream')

  DO it=1,31
    DO i=1,1000
      OPEN(11, file='latlon.txt', FORM='formatted', STATUS='old')
      READ(11, '(a6,7xf7.3,1x,f7.3)', iostat=istat1) stnid, lon, lat
      IF (istat1 < 0) EXIT

      time = 0.
      nlev = 1
      nflag = 1

      OPEN(12, file='asstn.txt', FORM='formatted', STATUS='old')
      DO j=1,20000
        READ(12, '(21x,a6,8x,i8,2x,f6.1,88x,f5.2)', iostat=istat2) id, date, temp, prcp
        IF (istat2 < 0) EXIT

        IF(id .eq. stnid .and. date .eq. (it+20101200)) THEN
          ! write the start header
          WRITE(21) id, lat, lon, time, nlev, nflag
          ! write the variables
          WRITE(21) prcp, temp
          GOTO 91
        END IF

      END DO
      CLOSE(12)

      ! fill no data station
      WRITE(21) stnid, lat, lon, time, nlev, nflag
      WRITE(21) 99.99, 99.99

    91 END DO
    CLOSE(11)

    ! write the terminate tail
    nlev = 0
    WRITE(21) stnid, lat, lon, time, nlev, nflag
    PRINT *,i-1

  END DO

  STOP
END PROGRAM
