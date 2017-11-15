!------------------------------------------------------------------------------
   module inv_mat
!------------------------------------------------------------------------------
!
! history log :
!   2017-02-24  Ki-Hwan Kim   Start
!
!------------------------------------------------------------------------------
   implicit none
!
   contains
!------------------------------------------------------------------------------
!
!
!------------------------------------------------------------------------------
   recursive function det(n, mat) result(accum)
!------------------------------------------------------------------------------
!
!  abstract : determinant of a matrix
!
!------------------------------------------------------------------------------
   implicit none
!
   integer,                  intent(in   ) :: n
   real(8), dimension(n, n), intent(in   ) :: mat
!   
   real(8), dimension(n-1, n-1) :: submat
   real(8)                      :: accum
   integer                      :: j, sgn
!------------------------------------------------------------------------------
   if (n == 1) then
     accum = mat(1,1)
   else
     accum = 0.D0
     sgn = 1
!
     do j = 1, n
       if (mat(1,j) /= 0) then
         submat(1:n-1, 1:j-1) = mat(2:n, 1:j-1)
         submat(1:n-1, j:n-1) = mat(2:n, j+1:n)
!
         accum = accum + sgn*mat(1,j)*det(n-1, submat)
       end if
       sgn = - sgn
     end do
   end if
   end function det
!------------------------------------------------------------------------------
!
!
!------------------------------------------------------------------------------
   function inv(n, row1, row2, mat) result(invmat)
!------------------------------------------------------------------------------
!
!  abstract : inverse matrix using the adjoint method
!
!------------------------------------------------------------------------------
   implicit none
!
   integer,                  intent(in   ) :: n, row1, row2
   real(8), dimension(n, n), intent(in   ) :: mat
!   
   real(8), dimension(n-1, n-1) :: submat
   real(8), dimension(n, n)     :: comat
   real(8), dimension(n, n)     :: invmat
   real(8)                      :: detmat
   integer                      :: i, j
!------------------------------------------------------------------------------
   detmat = det(n, mat)
!
   do j = row1, row2
     do i = 1, n
       submat(1:i-1, 1:j-1) = mat(1:i-1, 1:j-1)
       submat(1:i-1, j:n-1) = mat(1:i-1, j+1:n)
       submat(i:n-1, 1:j-1) = mat(i+1:n, 1:j-1)
       submat(i:n-1, j:n-1) = mat(i+1:n, j+1:n)
!
       invmat(j,i) = (-1)**(i+j)*det(n-1, submat)/detmat
     end do
   end do
   end function inv
!------------------------------------------------------------------------------
!
!
!------------------------------------------------------------------------------
   end module inv_mat
