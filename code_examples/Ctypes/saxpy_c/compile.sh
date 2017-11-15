#!/bin/sh

gcc -O3 -W -Wall -shared -fPIC -o saxpy.so saxpy.c
