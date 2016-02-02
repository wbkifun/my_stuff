#!/bin/bash
#PBS -l select=1:ncpus=20
#PBS -q normal20
#PBS -N remap_120
#PBS -j oe
#PBS -V


cd /home/khkim/usr/lib/python/util/remap/
#mpirun -np 32 -machinefile $PBS_NODEFILE time python cube_remap_matrix.py --rotated 30 180x360 regular ll2cs vgecore /nas2/user/khkim/remap_matrix/ >& run_cube_remap.log

#mpirun -np 20 python cube_remap_matrix.py --rotated 120 768x1024 regular cs2ll lagrange ./remap_matrix/ >& ne120_rotated_768x1024_regular_cs2ll_lagrange.log
mpirun -np 20 python cube_remap_matrix.py --rotated 120 768x1024 regular cs2ll vgecore ./remap_matrix/ >& ne120_rotated_768x1024_regular_cs2ll_vgecore.log
