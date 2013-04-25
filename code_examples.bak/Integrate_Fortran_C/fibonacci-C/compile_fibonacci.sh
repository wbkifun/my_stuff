#!/bin/bash

if [ -f 'fib1.so' ]; then 
	rm fib1.so
fi

gcc -O3 -fpic -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include --shared fib1.c -o fib1.so
