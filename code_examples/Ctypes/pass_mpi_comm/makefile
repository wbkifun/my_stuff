.PHONY : clean

FC = mpif90
FCFLAGS = -O3 -W -Wall

whoami.so : whoami.o
	$(FC) -shared -o whoami.so whoami.o

whoami.o : whoami.f90
	$(FC) $(FCFLAGS) -c -fPIC whoami.f90 -o whoami.o

clean :
	rm -f whoami.so whoami.o
