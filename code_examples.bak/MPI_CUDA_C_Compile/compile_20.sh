rm 20.exe mpi_20_cuda.o cufunc_20.o

gcc -I/usr/include/lam -I. -c mpi_20_cuda.c
nvcc -c cufunc_20.cu
gcc -llam -L/usr/local/cuda/lib -lcudart mpi_20_cuda.o cufunc_20.o -o 20.exe
