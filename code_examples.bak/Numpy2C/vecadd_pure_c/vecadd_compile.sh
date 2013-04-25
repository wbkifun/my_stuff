#!/bin/sh

gcc -O3 -std=c99 -fpic -shared -fopenmp -msse vecadd.c -o vecadd.so
