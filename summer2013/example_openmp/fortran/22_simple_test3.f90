program simple_test3
    integer, parameter :: N = 4
    integer :: i, tid, omp_get_thread_num

    call omp_set_num_threads(4)
!$omp parallel private(i, tid)
    tid = omp_get_thread_num()
    if( tid /= 2 ) then
        call sleep(2)
    end if

    !$omp sections
    !$omp section
    do i=0, N-1
        print *, 'L1 tid=',tid
    end do
    !$omp section
    do i=0, N-1
        print *, 'L2 tid=',tid
    end do
    !$omp end sections
!$omp end parallel
end
