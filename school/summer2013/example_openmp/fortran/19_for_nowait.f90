program nowait_for
    integer, parameter :: N = 9999
    integer i, j, a(0:N, 0:N)

    call omp_set_num_threads(4)
!$omp parallel private(i,j)
    !$omp do
    do j=0, N
        do i=0, j
            a(i,j) = i+j
        end do
    end do
    !$omp end do nowait

    !$omp do
    do j=0, N
        do i=j+1, N
            a(i,j) = i-j
        end do
    end do
!$omp end parallel
end
