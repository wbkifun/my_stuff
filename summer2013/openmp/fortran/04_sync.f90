program sync
	implicit none
    integer(8) :: i, sum=0

!$omp parallel
    !$omp do
    do i=1, N
        !$omp atomic
        sum = sum + i
    end do
!$omp end parallel

    print *, 'sum =', sum
end
