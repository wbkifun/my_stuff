#!/bin/sh

gfortran -O3 -c -fPIC inv_mat.f90
f2py -c --fcompiler=gnu95 --f90flags='-O3' -I. inv_mat.o -m inv_mat_py inv_mat_py.f90 
