program task_data_scope
    integer :: a=1, b=2, c=3, d=4, e=5
    integer tid, omp_get_thread_num

    call omp_set_num_threads(4)
!$omp parallel private(b,d,e,tid)
    tid = omp_get_thread_num()
    print 10, 'tid=',tid, ' a=',a, ' b=',b, ' c=',c, ' d=',d, ' e=',e
    !$omp single
    a=2; b=3; c=4; d=5;e=6
    !$omp end single
    !$omp task private(e)
    print 10, 'task tid=',omp_get_thread_num(), &
              &  ' a=',a, ' b=',b, ' c=',c, ' d=',d, ' e=',e
    !$omp end task
!$omp end parallel

10 format(A,I4,A,I4,A,I4,A,I4,A,I4,A,I4)
end
