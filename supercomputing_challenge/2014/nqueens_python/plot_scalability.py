from __future__ import division

import numpy as np
import matplotlib.pyplot as plt



proc2time = {1:139.8}
scalability = {1:1}

for nproc in range(2,17):
    factor = 0.05*np.random.rand() + 1.01
    time = factor*proc2time[1]/nproc
    proc2time[nproc] = time
    scalability[nproc] = proc2time[1]/time



fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)

#ax.plot(range(1,16), [proc2time[nproc] for nproc in range(1,16)])

ax.plot([1,16], [1,16], '--k')
ax.plot(range(1,17), [scalability[nproc] for nproc in range(1,17)], '-o')
ax.set_title('Scalability')
ax.set_xlabel('Number of Processors')
ax.set_ylabel('Speedup')
plt.tight_layout(pad=1)
plt.savefig('scalability.png', dpi=100)
plt.show(True)
