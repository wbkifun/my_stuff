#!/bin/sh

gcc -c func.c
gfortran main.f90 func.o -o a.out
