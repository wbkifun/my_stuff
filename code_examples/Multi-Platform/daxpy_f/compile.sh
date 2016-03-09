#!/bin/bash

SOFILE=tmpAVJE.so

if [ -f $SOFILE ];
then
    rm tmpAVJE.so
fi


module purge
module load anaconda2 gcc/4.7.1
f2py -c --fcompiler=gnu95 daxpy_f.pyf daxpy.f90

#module purge
#module load anaconda2 intel/2013
#f2py -c --fcompiler=intelem daxpy_f.pyf daxpy.f90


python main.py
