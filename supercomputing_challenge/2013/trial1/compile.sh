#!/bin/bash

if [ -f core_fortran.so ]; then
    rm core_fortran.so
fi
f2py -c --fcompiler=gnu95 -m core_fortran wave2d.f90
