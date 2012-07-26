#!/bin/sh

nvcc -I/usr/lib/openmpi/include -lmpi 140-vecadd-mpi.cu -o 140-vecadd-mpi.exe

