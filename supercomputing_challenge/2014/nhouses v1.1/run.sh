#!/bin/sh

#gcc -O3 serial.c -o serial
#thorq --add --mode single --base-dir here --name serial_c serial

#gfortran -O3 serial.f90 -o serial_f
#thorq --add --mode single --base-dir here --name serial_f serial_f

mpicc -O3 parallel.c -o parallel
thorq --add --mode mpi --node 1 --slots 2 --base-dir here --name p02 parallel
thorq --add --mode mpi --node 1 --slots 3 --base-dir here --name p03 parallel
thorq --add --mode mpi --node 1 --slots 4 --base-dir here --name p04 parallel
thorq --add --mode mpi --node 1 --slots 8 --base-dir here --name p08 parallel
thorq --add --mode mpi --node 1 --slots 16 --base-dir here --name p16 parallel
thorq --add --mode mpi --node 2 --slots 16 --base-dir here --name p32 parallel


watch --interval 1 thorq --stat-all
