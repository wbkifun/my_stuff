#!/bin/bash

#gcc -I/usr/local/cuda/include -lOpenCL 010-vecadd-opencl.c -o 010-veccadd-opencl.exe
#gcc -I/usr/local/cuda/include -lOpenCL 020-vec_add_sub.c -o 020-vec_add_sub.exe
gcc -I/usr/local/cuda/include -lOpenCL 030-vec_add_sub-multi.c -o 030-vec_add_sub-multi.exe

