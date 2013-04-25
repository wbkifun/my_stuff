#!/usr/bin/env python


# Read the text file
import numpy as np
import sys
try:
	fpath = sys.argv[1]
except IndexError:
	print('Error: file name required.')
	sys.exit()

f = open(fpath, 'r')
lines = f.readlines()
f.close()
nx = len(lines)

freq = np.zeros(nx)
amp = np.zeros(nx)

for i, line in enumerate(lines):
	datas = line.split()
	freq[i] = datas[0]
	amp[i] = datas[1]


# Plot
from matplotlib.pyplot import *
plot(freq, amp, 'x-')
xlabel('Wavelength (nm)')
ylabel(r'Amplitude ($\mu$w)')
xlim(freq[0], freq[-1])
ylim(min(amp), max(amp))
savefig('amp.png', dpi=150)
show()
