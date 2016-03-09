#!/bin/bash

SOFILE=tmpAVJE.so

if [ -f $SOFILE ];
then
    rm tmpAVJE.so
fi


module purge
module load anaconda2 gcc/4.7.1
f2py -c --compiler=unix daxpy_c.pyf daxpy.c

#module purge
#module load anaconda2 intel/2013
#f2py -c --compiler=intelem daxpy_c.pyf daxpy.c


python main.py
