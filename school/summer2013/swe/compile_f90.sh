#!/bin/bash

if [ -f sem_cores.so ]; then
	rm sem_cores.so
fi

f2py -c --fcompiler=gnu95 -m sem_cores sem_cores.f90
