#!/usr/bin/env python

import subprocess as sp

cmd = 'ssh 163.152.45.251 nvidia-smi -a'

stdout, stderr = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
print 'stdout', stdout
print stderr
