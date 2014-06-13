program simple_test4
    integer, parameter :: N = 4
    integer :: i, tid, omp_get_thread_num

    call omp_set_num_threads(4)
!$omp parallel private(i, tid)
    tid = omp_get_thread_num()
    !$omp sections
    !$omp section
    do i=0, N-1
        print *, 'L1 tid=',tid
    end do
    !$omp section
    do i=0, N-1
        print *, 'L2 tid=',tid
    end do
    call sleep(2);
    !$omp end sections
    print *, 'end tid=', tid
!$omp end parallel
end
