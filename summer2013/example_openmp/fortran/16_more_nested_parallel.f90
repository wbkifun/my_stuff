program more_nested_parallel
    integer x, tid, level
    integer omp_get_thread_num
    integer omp_get_level
    integer omp_get_ancestor_thread_num

    call omp_set_nested(.true.)
    call omp_set_num_threads(4)
!$omp parallel private(tid, level)
    tid = omp_get_thread_num()
    level = omp_get_level()
    print 10, 'thread id =', tid
    if( tid == 1 ) then
        !$omp parallel private(tid) num_threads(tid+2)
        tid = omp_get_thread_num()
        print 20, 'thread id =', tid, ' ancestor_thread_num(',level,')=',omp_get_ancestor_thread_num(level)
        !$omp end parallel
    end if
!$omp end parallel

10 format(A,I4)
20 format(T8,A,I4,A,I4,A,I4)
end
