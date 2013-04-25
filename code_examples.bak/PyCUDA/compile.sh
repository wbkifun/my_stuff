#!/bin/sh

if [ -f wave2d_*.so ]; then
	rm wave2d_*.so
fi

#gcc -O3 -fpic -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include --shared wave2d_cfunc.c -o wave2d_cfunc.so

#gcc -O3 -fpic -msse -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include --shared wave2d_cfunc-sse.c -o wave2d_cfunc.so

gcc -O3 -fpic -msse -fopenmp -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include --shared wave2d_cfunc-sse-openmp.c -o wave2d_cfunc.so


f2py -m wave2d_frout -c wave2d_frout.f

f2py -m wave2d_frout -c --f77flags='-fopenmp -lgomp' wave2d_frout-openmp.f
