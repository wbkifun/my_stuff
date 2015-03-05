#!/bin/bash
#rm matmul.o ext_matmul.so
#gcc -O3 -fPIC -g -I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include -c matmul.c -o matmul.o
#gcc -shared -o ext_matmul.so matmul.o

rm arr1d.o ext_arr1d.so
#gcc -O3 -fPIC -g -I/usr/include/python2.5 -I/usr/lib/python2.5/site-packages/numpy/core/include -c arr1d.c -o arr1d.o
#gcc -O3 -fPIC -g -I/usr/include/python2.7 -I/usr/lib/python2.7/dist-packages/numpy/core/include -c arr1d.c -o arr1d.o
gcc -O3 -fPIC -g -I/usr/include/python2.7 -c arr1d.c -o arr1d.o
gcc -shared -o ext_arr1d.so arr1d.o
