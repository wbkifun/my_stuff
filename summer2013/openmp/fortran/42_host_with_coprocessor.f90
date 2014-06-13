program host_with_coprocessor
    use omp_lib
    implicit none

!$omp parallel
    !$omp sections
    !$omp section
        !dir$ offload begin target(mic:0)
        !$omp parallel
            print *, 'hello ', omp_get_thread_num()
        !$omp end parallel
        !dir$ end offload

    !$omp section
        !dir$ offload begin target(mic:1)
        !$omp parallel
            print *, 'hello ', omp_get_thread_num()
        !$omp end parallel
        !dir$ end offload
    !$omp end sections
!$omp end parallel
end
