#!/usr/bin/env python

import sys
import numpy as np


def gaussian(x, sigma, x0=0):
	return (1. / (np.sqrt(2 * np.pi) * sigma ))* np.exp(- 0.5 * (np.float32(x - x0) / sigma)**2 )


dx = 0.01
nx = 1024
x = np.arange(nx) * dx
psi = np.zeros(nx, dtype=np.complex64)
psi.real[:] = gaussian(x, sigma=20*dx, x0=200*dx)
psi[:] *= np.exp(1j * 20 * x)

import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

l1, = ax1.plot(x, np.abs(psi), color='black', linewidth=2)
l2, = ax1.plot(x, psi.real, color='red')
l3, = ax1.plot(x, psi.imag, color='blue')
ax1.set_xlim(x[0], x[-1])

k = (np.fft.fftfreq(nx, dx)[:] * 2 * np.pi)
k_shift = np.fft.fftshift(k)
kpsi = np.fft.fft(psi)
kpsi_shift = np.fft.fftshift(kpsi)
l1, = ax2.plot(k_shift, np.abs(kpsi_shift), color='black', linewidth=2)
l2, = ax2.plot(k_shift, kpsi_shift.real, color='red')
l3, = ax2.plot(k_shift, kpsi_shift.imag, color='blue')
ax2.set_xlim(k_shift[nx/2 - 100], k_shift[nx/2 + 100])

#plt.show()
plt.savefig('gaussian_function.png')
