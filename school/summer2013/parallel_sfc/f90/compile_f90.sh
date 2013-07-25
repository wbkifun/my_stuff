#!/bin/bash

if [ -f interact_between_elems.so ]; then
	rm interact_between_elems.so
fi

if [ -f interact_between_elems_nolev.so ]; then
	rm interact_between_elems_nolev.so
fi

f2py -c --fcompiler=gnu95 -m interact_between_elems interact_between_elems.f90
f2py -c --fcompiler=gnu95 -m interact_between_elems_nolev interact_between_elems_nolev.f90
