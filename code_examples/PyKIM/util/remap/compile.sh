#!/bin/bash

f2py -c --fcompiler=gnu95 -m cube_remap_core cube_remap_core.f90
