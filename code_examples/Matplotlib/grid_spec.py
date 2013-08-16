from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


data = np.arange(100).reshape(10,10)

fig = plt.figure(figsize=(12,9))    # width, height in inches
gs = GridSpec(nrows=3, ncols=4, hspace=0)
ax1 = plt.subplot(gs[1,1])
ax2 = plt.subplot(gs[1,2])
ax3 = plt.subplot(gs[1,3])
ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[2,1])
ax6 = plt.subplot(gs[0,1])

ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
for ax in ax_list:
    ax.imshow(data)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
