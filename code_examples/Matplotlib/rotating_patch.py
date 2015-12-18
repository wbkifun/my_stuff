import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl



fig = plt.figure()
ax = fig.add_subplot(111)

rect = patches.Rectangle((0.0120,0),0.1,1000)

t_start = ax.transData
t = mpl.transforms.Affine2D().rotate_deg(-45)
t_end = t_start + t

rect.set_transform(t_end)

print repr(t_start)
print repr(t_end)
ax.add_patch(rect)

plt.show()
