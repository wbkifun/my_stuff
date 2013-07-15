program task2
    integer omp_get_thread_num

!$omp parallel num_threads(32)
    !$omp single
    print *, 'A tid=', omp_get_thread_num()
    print *, 'B tid=', omp_get_thread_num()
    print *, 'C tid=', omp_get_thread_num()
    print *, 'D tid=', omp_get_thread_num()
    !$omp end single
!$omp end parallel
end
