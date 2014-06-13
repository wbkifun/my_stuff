program nested_parallel
    integer tid, omp_get_thread_num

    call omp_set_nested(1)
    call omp_set_num_threads(2)
!$omp parallel private(tid)
    tid = omp_get_thread_num()
    print 10, 'thread id =', tid
    if( tid == 1 ) then
        !$omp parallel private(tid)
        tid = omp_get_thread_num()
        print 20, 'thread id =', tid
        !$omp end parallel
    end if
!$omp end parallel

10 format(A,I4)
20 format(T8,A,I4)
end
