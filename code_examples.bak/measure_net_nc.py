import numpy as np
import subprocess as sp

from mpi4py import MPI
from scipy import optimize
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def fit_bandwidth_latency(nbytes, dts):
	fitfunc = lambda p, x: p[0] * x + p[1]
	errfunc = lambda p, x, y: fitfunc(p, x) - y

	p0 = np.array([1e3, 0])
	p1, success = optimize.leastsq(errfunc, p0, args=(nbytes, dts))
	bandwidth = 1. / p1[0]
	latency = p1[1]

	return (bandwidth, latency)


# Main
nys = range(128, 512, 16)
nbytes = np.zeros(len(nys), np.int64)
dts = np.zeros(len(nys))

target_send = rank-1 if rank > 0 else size-1
target_recv = rank+1 if rank < size-1 else 0

for i, ny in enumerate(nys):
    shape = (2, ny, 256)
    arr_send = np.random.rand(*shape).astype(np.float32)
    arr_recv = np.zeros_like(arr_send)

    nbytes[i] = arr_send.nbytes * 2


    req_send = comm.Send_init(arr_send, target_send, 0)
    req_recv = comm.Recv_init(arr_recv, target_recv, 0)

    
    tmax = 10
    t0 = time()
    for j in range(tmax):
        req_send.Start()
        req_recv.Start()
        req_send.Wait()
        req_recv.Wait()

    dts[i] = (time() - t0) / tmax

    if rank == 0: print i 

bw, lt = fit_bandwidth_latency(nbytes,dts)
print(rank, 'bandwidth: %g' % bw)
print(rank, 'latency: %g' % lt)
'''


# Plot
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
#plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
xticks = range(8, 33)

mdts = dts * 1e3
fit_dt = lambda x, bw, lt: (1./bw * x + lt) * 1e3
p0 = ax1.plot(xticks, mdts, linestyle='None', color='k', marker='o')#, markersize=5)
p1 = ax1.plot(xticks, fit_dt(nbyte, bw, lt), color='k')
ax1.set_xlabel(r'Size [$\times2^{16}$ nbyte]')
ax1.set_ylabel(r'Time [ms]')
ax1.set_xlim(7, 33)
ax1.set_ylim(mdts.min()*0.9, mdts.max()*1.1)
ax1.legend((p0, p1), ('Measure', 'Fitted'), loc='best', numpoints=1)
plt.savefig('measure.eps', dpi=150)
plt.show()
'''
