#!/bin/sh

if [ -f 'hw_f.so' ]; then 
	rm hw_f.so
fi

if [ -f 'hw.pyf' ]; then 
	f2py -c --fcompiler=gnu95 hw.pyf hw.f90
else
	f2py -m hw_f -c --fcompiler=gnu95 hw.f90
fi
