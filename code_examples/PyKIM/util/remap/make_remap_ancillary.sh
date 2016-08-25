#!/bin/bash
#PBS -l select=1:ncpus=16
#PBS -q normal
#PBS -N remap_ancil
#PBS -j oe
#PBS -V


module purge
module load PrgEnv-gnu/4.7.1 anaconda2
#export PATH="/home/khkim/anaconda2/bin:$PATH"

export WORKDIR=/home/khkim/usr/lib/python/util/remap

NE=30
CS_TYPE="regular"
REMAP_MATRIX_DIR="/data/KIM2.3/remap_matrix/"
SRC_DIR="/data/KIM2.3/ancillary_preprocessed/"
DST_DIR=$WORKDIR/ancillary

mpirun -np 16 $WORKDIR/remap_ancillary.py $NE $CS_TYPE $REMAP_MATRIX_DIR $SRC_DIR $DST_DIR
