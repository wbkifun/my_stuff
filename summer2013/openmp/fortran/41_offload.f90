program sum_offload
    use omp_lib
    implicit none
    integer(8) :: i, total_sum = 0

!dir$ offload begin target(mic)
!$omp parallel do reduction (+:total_sum)
    do i=1,N
        total_sum = total_sum + i
    end do
!dir$ end offload

    print *, 'sum = ', total_sum
end
