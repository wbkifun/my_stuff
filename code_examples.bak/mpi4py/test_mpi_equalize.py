from __future__ import division
import numpy
import subprocess as sp


def test_equalize():
    cmd = 'mpirun -n 5 python mpi_equalize.py'
    #cmd = 'python mpi_equalize.py'
    proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    #print stdout
    assert 'assert' not in stderr, stderr
