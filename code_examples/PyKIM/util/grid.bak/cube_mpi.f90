SUBROUTINE pre_send(ep_size, map_size, send_buf_size, recv_buf_size, local_src_size, dsts, srcs, wgts, f, send_buf, recv_buf)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: ep_size, map_size, send_buf_size, recv_buf_size, local_src_size
  INTEGER, INTENT(IN) :: dsts(map_size), srcs(map_size)
  REAL(8), INTENT(IN) :: wgts(map_size)
  REAL(8), INTENT(IN) :: f(ep_size)
  REAL(8), INTENT(INOUT) :: send_buf(send_buf_size), recv_buf(recv_buf_size)

  INTEGER :: i, dst, src
  REAL(8) :: wgt

  send_buf(:) = 0
  recv_buf(:) = 0

  DO i=1,map_size
    dst = dsts(i) + 1
    src = srcs(i) + 1
    wgt = wgts(i)
    IF (i <= local_src_size) THEN
      recv_buf(dst) = recv_buf(dst) + wgt*f(src)
    ELSE
      send_buf(dst) = send_buf(dst) + wgt*f(src)
    END IF
  END DO
END SUBROUTINE




SUBROUTINE post_recv(ep_size, map_size, recv_buf_size, dsts, srcs, recv_buf, f)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: ep_size, map_size, recv_buf_size
  INTEGER, INTENT(IN) :: dsts(map_size), srcs(map_size)
  REAL(8), INTENT(IN) :: recv_buf(recv_buf_size)
  REAL(8), INTENT(INOUT) :: f(ep_size)

  INTEGER :: i, dst, src, prev_dst

  prev_dst = -1
  DO i=1,map_size
    dst = dsts(i) + 1
    src = srcs(i) + 1

    IF (prev_dst /= dst) THEN
      f(dst) = 0
      prev_dst = dst
    END IF

    f(dst) = f(dst) + recv_buf(src)
  END DO
END SUBROUTINE
