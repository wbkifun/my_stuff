program mat_mul_dynamic
    integer i,j,k, a(0:N-1,0:N-1), b(0:N-1,0:N-1), c(0:N-1,0:N-1)

    c = 0
    do j=0, N-1
        do i=0,j
            a(i,j) = i+j+1
            b(i,j) = i+j+2
        end do
        do i=j+1,N-1
            a(i,j) = 0
            b(i,j) = 0
        end do
    end do

!$omp parallel do private(i,k) &
!$omp& schedule(dynamic,5)
    do j=0, N-1
        do k=0, j
            do i=k, j
                c(i,j) = c(i,j) + a(i,k)*b(k,j)
            end do
        end do
    end do
!$omp end parallel do

    print *, 'data = ', c(N-1,N-1)
end
