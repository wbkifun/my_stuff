.PHONY : clean

FC = gfortran
FCFLAGS = -O3 -W -Wall

saxpy.so : saxpy.o
	$(FC) -shared -o saxpy.so saxpy.o

saxpy.o : saxpy.f90
	$(FC) $(FCFLAGS) -c -fPIC saxpy.f90 -o saxpy.o

clean :
	rm -f saxpy.so saxpy.o
