program first_touch
    use omp_lib
    implicit none
    integer(8) :: i
    double precision :: A(N)

!$omp parallel do
    do i=1, N
        A(i) = 0.0
    end do
!$omp end parallel do

!$omp parallel do
    do i=1, N
        !......
    end do
!$omp end parallel do

end
