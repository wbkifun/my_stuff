SUBROUTINE my_cos(nx, in_arr, out_arr)
  INTEGER, INTENT(IN) :: nx
  REAL(8), DIMENSION(nx), INTENT(IN) :: in_arr
  REAL(8), DIMENSION(nx), INTENT(INOUT) :: out_arr

  INTEGER :: i

  DO i=1, nx
    out_arr(i) = cos(in_arr(i))
  END DO
END SUBROUTINE



SUBROUTINE my_cos_2d(nx, ny, in_arr, out_arr)
  INTEGER, INTENT(IN) :: nx, ny
  REAL(8), DIMENSION(nx, ny), INTENT(IN) :: in_arr
  REAL(8), DIMENSION(nx, ny), INTENT(INOUT) :: out_arr

  INTEGER :: i, j

  DO j=1, ny
    DO i=1, nx
      out_arr(i, j) = cos(in_arr(i, j))
    END DO
  END DO
END SUBROUTINE
