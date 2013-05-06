#!/bin/sh

nvcc -arch=sm_20 -c func.cu
gfortran -lcuda -lcudart main.f90 func.o -o main
