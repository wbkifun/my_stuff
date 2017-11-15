.PHONY : clean

FC = gfortran
FCFLAGS = -O3 -W -Wall

my_math.so : my_math.o
	$(FC) -shared -o my_math.so my_math.o

my_math.o : my_math.f90
	$(FC) $(FCFLAGS) -c -fPIC my_math.f90 -o my_math.o

clean :
	rm -f my_math.so my_math.o
