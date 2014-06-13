program task1

!$omp parallel num_threads(32)
    write(*, '(A)', advance="no") 'A '
    write(*, '(A)', advance="no") 'B '
    write(*, '(A)', advance="no") 'C '
    write(*, '(A)', advance="no") 'D '
    write(*, *) ' '
!$omp end parallel
end
