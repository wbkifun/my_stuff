program test
implicit none

integer, parameter :: n = 1000000
integer i
real s, a(n)

do i =1, n
a(i) = real(i)
enddo

!$OMP PARALLEL DO REDUCTION(+:s)
do i =1, n
  s = s + a(i)
enddo

end program test
