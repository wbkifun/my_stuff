program create_thread
    use omp_lib
    implicit none

    print *, 'threads = ', omp_get_num_threads()
!$omp parallel num_threads(3)
    print *, 'tid = ', omp_get_thread_num(), ' threads = ', omp_get_num_threads()
!$omp end parallel
    print *, 'threads = ', omp_get_num_threads()
!$omp parallel
    print *, 'tid = ', omp_get_thread_num(), ' threads = ', omp_get_num_threads()
!$omp end parallel
    call omp_set_num_threads(4)
    print *, 'threads = ', omp_get_num_threads()
!$omp parallel
    print *, 'tid = ', omp_get_thread_num(), ' threads = ', omp_get_num_threads()
!$omp end parallel
    print *, 'threads = ', omp_get_num_threads()
end
