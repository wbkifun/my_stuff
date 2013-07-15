program simple_test1
    integer, parameter :: N = 20
    integer :: i, tid, a(N), omp_get_thread_num

    call omp_set_num_threads(4)
!$omp parallel private(tid)
    tid = omp_get_thread_num()
    if( tid /= 2 ) then
        call sleep(2)
    end if
    !$omp do
    do i=0, N-1
        a(i) = i
        print *, 'a(',i,')=',a(i),' tid=',tid
    end do
    !$omp end do
    print *, 'end ',tid,' thread'
!$omp end parallel
end
