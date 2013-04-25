rm mpi_02_cuda
nvcc -I/usr/include/lam -llam mpi_02_cuda.cu -o 02.exe
