#!/bin/sh

gcc -O3 -std=c99 -fpic -fopenmp -msse -I/usr/include/python2.7 vecop.c -shared -o vecop.so
