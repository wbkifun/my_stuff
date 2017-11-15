subroutine inv_f90(n, row1, row2, mat, invmat)
!------------------------------------------------------------------------------
!
! Wrapper of inv_mat.f90 for Python test
!
!------------------------------------------------------------------------------
use inv_mat, only: inv
implicit none

integer,                  intent(in   ) :: n, row1, row2
real(8), dimension(n, n), intent(in   ) :: mat
real(8), dimension(n, n), intent(inout) :: invmat
 
invmat = inv(n, row1, row2, mat)
end subroutine inv_f90
