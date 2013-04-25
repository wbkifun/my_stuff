#!/bin/sh

#nvcc -Xcompiler -fopenmp gfree.cu -o gfree

#gcc -O3 -std=c99 -fopenmp -lgomp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart gfree.c -o gfree

gcc -O3 -std=c99 -fopenmp -lgomp -I/usr/local/cuda/include -lcuda gfree.c -o gfree

sudo cp gfree /usr/bin/
sudo cp gfree /nfsroot/usr/bin/
