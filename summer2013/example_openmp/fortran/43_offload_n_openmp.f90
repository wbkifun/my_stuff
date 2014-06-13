program sum_offload_n_openmp2
    use omp_lib
    implicit none
    integer(8) :: i, total_sum = 0

!$omp parallel
    !$omp single
        !dir$ offload begin target(mic)
            print *, 'check'
        !dir$ end offload
    !$omp end single

    !$omp do reduction (+:total_sum)
    do i=1,N
        total_sum = total_sum + i
    end do
    !$omp end do
!$omp end parallel
end
