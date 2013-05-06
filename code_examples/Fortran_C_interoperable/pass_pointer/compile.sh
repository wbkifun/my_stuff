#!/bin/sh

gfortran -c main.f90
gcc -c func.c
gfortran main.o func.o -o a.out
