program task3
    integer omp_get_thread_num

!$omp parallel num_threads(32)
    !$omp single
    print *, 'A tid=', omp_get_thread_num()
    !$omp task
    print *, 'B tid=', omp_get_thread_num()
    !$omp end task
    !$omp task
    print *, 'C tid=', omp_get_thread_num()
    !$omp end task
    print *, 'D tid=', omp_get_thread_num()
    !$omp end single
!$omp end parallel
end
