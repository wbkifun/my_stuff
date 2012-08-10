program test_modulo
    implicit none
    integer :: i

    print *, modulo(17,3)
    print *, modulo(17.5,5.5)

    print *, modulo(-17,3)
    print *, modulo(-17.5,5.5)

    print *, modulo(17,-3)
    print *, modulo(17.5,-5.5)

    print *, mod(17,3)
    print *, mod(17.5,5.5)

    print *, mod(-17,3)
    print *, mod(-17.5,5.5)

    print *, mod(17,-3)
    print *, mod(17.5,-5.5)

    do i=-10,10
      print *, i, i /= -1
    end do

    print *, 0, 0 /= -1
    print *, 249, 249 /= -1
    print *, 4, 4 /= -1
    print *, 253, 253 /= -1
    print *, -1, -1 /= -1
    print *, 8, 8 /= -1
    print *, 9, 9 /= -1
    print *, 257, 257 /= -1

end program
