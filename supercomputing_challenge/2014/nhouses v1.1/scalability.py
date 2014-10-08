from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


procs = [2,3,4,8,16,32]
n16 = [257.243, 129.032, 87.914, 40.278, 20.480, 10.591]
n17 = [2018.879, 1000.666, 681.054, 319.978, 156.621, 79.158]

n16_speedup = [n16[0]/t for t in n16]
n17_speedup = [n17[0]/t for t in n17]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([1,32], [1,32], 'k:', label='Ideal')
ax.plot(procs, n16_speedup, 'bo-', label='n16')
ax.plot(procs, n17_speedup, 'ro-', label='n17')

ax.set_xlim(1,32)

ax.set_xlabel('Number of processes')
ax.set_ylabel('Speedup')
ax.set_title('Scalability')

plt.legend(loc='lower right')
plt.tight_layout()
plt.show(True)
