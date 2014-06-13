program sum_offload_n_openmp3
    use omp_lib
    implicit none
    integer(8) :: i, total_sum = 0, local_sum

!$omp parallel private(local_sum)
    local_sum = 0
    !$omp single
        !dir$ offload begin target(mic)
            print *, 'check'
        !dir$ end offload
    !$omp end single nowait

    !$omp do schedule(dynamic)
    do i=1,N
        local_sum = local_sum + i
    end do
    !$omp end do

    !$omp atomic
    total_sum = total_sum + local_sum
!$omp end parallel
end
