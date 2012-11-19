#!/bin/bash

if [ -f core_fortran.so ]; then
    rm core_fortran.so
fi
f2py -c --fcompiler=gnu95 -m core_fortran core.f90


#if [ -f core.cubin ]; then
#    rm core.cubin
#fi
#nvcc -arch=sm_20 --cubin core.cu
