program reduction
	implicit none
    integer(8) :: i, sum=0

!$omp parallel do reduction (+:sum)
    do i=1, N
        sum = sum + i
    end do
!$omp end parallel do

    print *, 'sum =', sum
end
