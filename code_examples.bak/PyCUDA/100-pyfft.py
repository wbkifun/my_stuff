#!/usr/bin/env python

import sys
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pyfft.cuda import Plan

from scipy.constants import physical_constants
LENGTH = physical_constants['atomic unit of length']
TIME = physical_constants['atomic unit of time']
ENERGY = physical_constants['atomic unit of energy']


def gaussian(x, sigma, x0=0):
	return (1. / (np.sqrt(2 * np.pi) * sigma ))* np.exp(- 0.5 * (np.float32(x - x0) / sigma)**2 )


# initialize
snx = 1024	# sub_nx
nx = snx * 16
dx = 0.01
dt = 0.0001
tmax = 20000
tgap = 100
psi = np.zeros(nx, dtype=np.complex64)
kpsi = np.zeros(nx, dtype=np.complex64)
sx0 = nx/2 - snx/2
sx1 = nx/2 + snx/2

# initial wavefunction
sigma0 = 50 * dx
k0 = 20
x = np.arange(nx) * dx
psi.real[:] = gaussian(x, sigma=sigma0, x0=(sx0+200)*dx)
psi[:] *= np.exp(1j * k0 * x)

# cuda init
cuda.init()
ctx = cuda.Device(0).make_context()
strm = cuda.Stream()
plan = Plan(nx, dtype=np.complex64, context=ctx, stream=strm)
psi_gpu = gpuarray.to_gpu(psi)

# potential
vx0, vwidth = nx/2, 70
vmax = (k0 ** 2) / 2

# plot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.ion()
fig = plt.figure(figsize=(8,6))
fig.subplots_adjust(left=0.08, right=0.78, hspace=0.3)
fig.suptitle('Finite Barrier', fontsize=18)
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.set_xlabel('x')
ax2.set_xlabel('k')
ax3.set_xlabel('k')
ax3.set_ylabel('Transmittance (a.u.)')

gmax = np.abs(psi).max() * 1.0
barr = Rectangle((vx0*dx, 0), vwidth*dx, gmax*0.9, facecolor='green', alpha=0.1)
ax1.add_patch(barr)
l1, = ax1.plot(x, np.abs(psi), color='black', linewidth=2)
l2, = ax1.plot(x, psi.real, color='red')
l3, = ax1.plot(x, psi.imag, color='blue')
ax1.set_xlim(sx0*dx, sx1*dx)
ax1.set_ylim(-gmax, gmax)
ax1.set_yticklabels([])
ax1.legend([barr, l1, l2, l3], ['V', r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

k = (np.fft.fftfreq(nx, dx)[:] * 2 * np.pi)
k_shift = np.fft.fftshift(k)
kpsi[:] = np.fft.fft(psi)
kpsi_shift = np.fft.fftshift(kpsi)
gmax_k = np.abs(kpsi).max() * 1.0
l21, = ax2.plot(k_shift, np.abs(kpsi_shift), color='black', linewidth=2)
l22, = ax2.plot(k_shift, kpsi_shift.real, color='red')
l23, = ax2.plot(k_shift, kpsi_shift.imag, color='blue')
ax2.set_xlim(-(k0 + 1/sigma0 * 5), k0 + 1/sigma0 * 5)
ax2.set_ylim(-gmax_k, gmax_k)
ax2.set_yticklabels([])
ax2.legend([l21, l22, l23], [r'$|\psi|$', r'$Re(\psi)$', r'$Im(\psi)$'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

# transmittance
analytic_t = np.zeros(nx, dtype=np.float32)
L = vwidth * dx
kidx = (k_shift < k0).argmin()
kappa = np.sqrt(k0**2 - k_shift[:kidx]**2)
xi = 0.5 * (kappa / k_shift[:kidx] - k_shift[:kidx] / kappa)
kL = kappa[:] * L
analytic_t[:kidx] = 1. / (np.cosh(kL)**2 + xi**2 * np.sinh(kL)**2)
kappa = np.sqrt(k_shift[kidx:]**2 - k0**2)
xi = 0.5 * (kappa / k_shift[kidx:] + k_shift[kidx:] / kappa)
kL = kappa[:] * L
analytic_t[kidx:] = 1. / (np.cos(kL)**2 + xi**2 * np.sin(kL)**2)

ax3.plot([k0, k0], [0, 1], linestyle='-.', color='black')
ax3.plot([k0, k_shift[-1]], [1, 1], linestyle='-.', color='black')
l30, = ax3.plot(k_shift, analytic_t, linestyle='--', color='black', linewidth=2)

kpsi_shift0 = kpsi_shift.copy()
l31, = ax3.plot(k_shift, (np.abs(kpsi_shift/kpsi_shift0))**2, color='blue', linewidth=2)
ax3.set_xlim(k0 - 1/sigma0*1, k0 + 1/sigma0*2)
ax3.set_ylim(0, 1.1)
ax3.legend([l30, l31], ['Analytic', 'Numeric'], bbox_to_anchor=(1.02,1), loc=2, borderaxespad=0.)

# time loop
k2 = k**2
lc = np.exp(- 0.5j * k2[:] * dt).astype(np.complex64)
lc_sqrt = np.exp(- 0.25j * k2[:] * dt).astype(np.complex64)
vc = np.exp(- 1j * vmax * dt).astype(np.complex64)

lc_gpu = gpuarray.to_gpu(lc)
lc_sqrt_gpu = gpuarray.to_gpu(lc_sqrt)
kernels = '''
	__global__ void vcf(float2 vc, float2 *psi) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		while ( tid < nx ) {
			psi[tid].x = vc.x * psi[tid].x - vc.y * psi[tid].y;
			psi[tid].y = vc.x * psi[tid].y + vc.y * psi[tid].x;
			tid += gridDim.x * blockDim.x;
		}
	}
'''
mod = SourceModule(kernels.replace('nx', str(nx)))
vcf = mod.get_function('vcf')

#kpsi[:] = np.fft.fft(psi[:])
#psi[:] = np.fft.ifft(np.exp(- 0.25j * k2[:] * dt) * kpsi[:])
#psi[vx0:vx0+vwidth] *= vc
plan.execute(psi_gpu)
print psi_gpu.dtype
print lc_sqrt_gpu.dtype
psi_gpu *= lc_sqrt_gpu
plan.execute(psi_gpu, inverse=True)
vcf(vc, psi_gpu, block=(256,1,1), grid=(nx/256,1))
"""
for tstep in xrange(tmax):
	kpsi[:] = np.fft.fft(psi[:])
	psi[:] = np.fft.ifft(lc[:] * kpsi[:])
	psi[vx0:vx0+vwidth] *= vc

	if tstep%tgap == 0:
		print "tstep = %d\r" % (tstep),
		sys.stdout.flush()

		l1.set_ydata(np.abs(psi))
		l2.set_ydata(psi.real * 1.0000001)
		l3.set_ydata(psi.imag * 1.0000001)

		kpsi_shift = np.fft.fftshift(kpsi)
		l21.set_ydata(np.abs(kpsi_shift))
		l22.set_ydata(kpsi_shift.real * 1.0000001)
		l23.set_ydata(kpsi_shift.imag * 1.0000001)
		l31.set_ydata((np.abs(kpsi_shift/kpsi_shift0))**2)

		#plt.savefig('./png/%.5d.png' % tstep, dpi=150)
		plt.draw()
psi[:] = np.fft.ifft(np.exp(- 0.25j * k2[:] * dt) * kpsi[:])
"""
ctx.pop()
