rm mpi_01 mpi_01_gcc
mpicc mpi_01.c -o mpi_01
gcc -I/usr/include/lam -llam mpi_01.c -o 01.exe
