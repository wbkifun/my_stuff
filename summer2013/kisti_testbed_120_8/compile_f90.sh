#!/bin/bash

if [ -f sem_cores.so ]; then
	rm sem_cores.so
fi
#f2py -c --fcompiler=gnu95 -m sem_cores sem_cores.f90


if [ -f interact_inner.so ]; then
	rm interact_inner.so
fi
#f2py -c --fcompiler=gnu95 -m interact_inner interact_inner.f90

#f2py -c --fcompiler=intelem --f90flags='-vec-report=2' -m sem_cores sem_cores.f90
#f2py -c --fcompiler=intelem --f90flags='-vec-report=2' -m interact_inner interact_inner.f90

#f2py -c --fcompiler=gnu95 --f90flags='-O3' -m sem_cores sem_cores.f90
#f2py -c --fcompiler=gnu95 --f90flags='-O3' -m interact_inner interact_inner.f90

f2py -c --fcompiler=intelem --f90flags='-no-vec' -m sem_cores sem_cores.f90
f2py -c --fcompiler=intelem --f90flags='-no-vec' -m interact_inner interact_inner.f90

#f2py -c --fcompiler=intelem -m sem_cores sem_cores.f90
#f2py -c --fcompiler=intelem -m interact_inner interact_inner.f90

#f2py -c --fcompiler=intelem --f90flags='-parallel' -m sem_cores sem_cores.f90
#f2py -c --fcompiler=intelem --f90flags='-parallel' -m interact_inner interact_inner.f90

#f2py -c --fcompiler=intelem --f90flags='-openmp' -m sem_cores sem_cores.f90
#f2py -c --fcompiler=intelem --f90flags='-openmp' -m interact_inner interact_inner.f90

#f2py -c --fcompiler=intelem --f90flags='-Ofast' -m sem_cores sem_cores.f90
#f2py -c --fcompiler=intelem --f90flags='-Ofast' -m interact_inner interact_inner.f90
