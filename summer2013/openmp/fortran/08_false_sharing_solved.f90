program false_sharing_solved
    integer(8) :: total_sum=0, local_sum(0:(TN-1)*8)
    integer i,tid, omp_get_thread_num

!$omp parallel private(tid) num_threads(TN)
    tid = omp_get_thread_num() * 8
    local_sum(tid) = 0
    !$omp do
    do i=1, N
        local_sum(tid) = local_sum(tid) + i
    end do
!$omp end parallel

    do i=0, TN-1
        total_sum = total_sum + local_sum(i*8)
    end do

    print *, 'sum=', total_sum
end
