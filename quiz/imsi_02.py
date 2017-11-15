import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi/4, np.pi/4, 100, endpoint=True)
#x=np.arange(0, 360. )

s1=np.sin(x-np.pi/2)
s2=np.sin(36*x-np.pi/2)
s3=np.sin(x-np.pi/2)+np.sin(36*x-np.pi/2)

ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2)
ax3 = plt.subplot(3,1,3)
ax1.plot(x,s1)
ax2.plot(x,s2)
ax3.plot(x,s3)

plt.show()

