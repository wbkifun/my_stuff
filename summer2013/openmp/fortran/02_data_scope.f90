program data_scope_firstprivate
    integer i, tid, omp_get_thread_num

    i = 10
    call omp_set_num_threads(4)
!$omp parallel private(tid) firstprivate(i)
    tid = omp_get_thread_num()
    print *, ' tid = ', tid, ' i = ', i
    i = 20
!$omp end parallel
    print *, ' tid = ', tid, ' i = ', i
end
