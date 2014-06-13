#!/bin/sh

#f2py -c --help-fcompiler


if [ -f 'LJ_03.so' ]; then
	rm LJ_03.so
fi

if [ -f 'LJ_03.pyf' ]; then
	rm LJ_03.pyf
fi

f2py -m LJ_03 -h LJ_03.pyf LJ_03_subroutines.f90 
f2py -c --fcompiler=intelem --f90flags='-Ofast' LJ_03.pyf LJ_03_subroutines.f90

#f2py -c --fcompiler=intelem LJ_03_subroutines.f90 -m LJ_03 
#f2py -c --fcompiler=gnu95 LJ_03_subroutines.f90 -m LJ_03
