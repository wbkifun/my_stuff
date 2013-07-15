program parallel_loop 
	implicit none
    integer(8) :: i, sum=0

!$omp parallel
    !$omp do
    do i=1, N
        sum = sum + i
    end do
!$omp end parallel

    print *, 'sum =', sum
end
