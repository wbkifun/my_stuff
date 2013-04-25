REAL FUNCTION hw1(a, b)
REAL a, b
hw1 = sin(a + b)
END FUNCTION hw1

SUBROUTINE hw2(a, b)
REAL a, b, s
s = sin(a + b)
WRITE(*,*) 'Hello, World! sin(', a, '+', b, ')=', s 
END SUBROUTINE hw2
