#!/bin/sh

#f2py -c --help-fcompiler


if [ -f 'LJ_04.so' ]; then
	rm LJ_04.so
fi

if [ -f 'LJ_04.pyf' ]; then
	rm LJ_04.pyf
fi

f2py -m LJ_04 -h LJ_04.pyf LJ_04_remove_if.f90 
f2py -c --fcompiler=intelem --f90flags='-Ofast' LJ_04.pyf LJ_04_remove_if.f90

#f2py -c --fcompiler=intelem LJ_04_remove_if.f90 -m LJ_04 
#f2py -c --fcompiler=gnu95 LJ_04_remove_if.f90 -m LJ_04
