program sections
    integer i, a(0:9), b(0:19)

    call omp_set_num_threads(2)

!$omp parallel
    !$omp sections
    !$omp section
    do i=0, 9
        a(i) = i * 10 + 5
    end do
    !$omp section
    do i=0, 19
        b(i) = i * 5 + 10
    end do
    !$omp end sections
!$omp end parallel

    write(*,10) a
    write(*,20) b

10 format(10i4)
20 format(20i4)
end 

