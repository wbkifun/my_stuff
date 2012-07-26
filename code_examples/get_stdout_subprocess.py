#!/usr/bin/env python

import subprocess as sp
import time

cmd = 'nvidia-smi -a'
while(True):
	ps = sp.Popen(cmd.split(), stdout=sp.PIPE)
	stdout, stderr = ps.communicate()
	print stdout
	time.sleep(1)

