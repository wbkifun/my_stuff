#!/bin/sh

if [ -f "field.bin" ]
then
	rm field.bin
fi

if [ -f "a.out" ]
then
	rm a.out
fi


#gfortran -O3 wave2d_circular.f90
gfortran -O3 wave2d_slit.f90
time ./a.out


#mpif90 -O3 wave2d_circular_mpi.f90
#mpif90 -O3 wave2d_slit_mpi.f90
#time mpirun -np 16 ./a.out
