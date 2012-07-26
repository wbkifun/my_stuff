#!/bin/sh
#-----------------------------------------------------------------------------
# File Name : compile.sh
#
# Author : Ki-Hwan Kim (wbkifun@korea.ac.kr)
# 
# Written date :	2010. 6. 17
# Modify date :		
#
# Copyright : GNU GPL
#
# Description : 
# Simulation for the 2-dimensional wave equations with simple FD (Finite-Difference) scheme
#
# These are educational codes to study python programming for high performance computing.
# Step 2-4: Combining with C function
#           Compile the C code for Python module
#-----------------------------------------------------------------------------

if [ -f wave2d_cfunc.so ]; then
	rm wave2d_cfunc.so
fi

#gcc -O3 -fpic -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include --shared wave2d-cfunc.c -o wave2d_cfunc.so

#gcc -O3 -fpic -msse -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include --shared wave2d-cfunc-sse.c -o wave2d_cfunc.so

gcc -O3 -fpic -msse -fopenmp -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include --shared wave2d-cfunc-sse-openmp.c -o wave2d_cfunc.so
