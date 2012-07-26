gcc -I/usr/include/lam -c mpi_10.c
gcc -c func_10.c
gcc -llam mpi_10.o func_10.o -o 10.exe
