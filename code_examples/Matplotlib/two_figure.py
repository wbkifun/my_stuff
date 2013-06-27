from __future__ import division
import matplotlib.pyplot as plt


fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)

ax1.plot([1,2,3], [4,5,6])
ax2.plot([1,2,3], [4.2,5.1,6.3])

plt.show()
