#!/bin/sh

nvcc -I/usr/lib/openmpi/include -lmpi doubling-mpi.cu -o doubling-mpi.exe

