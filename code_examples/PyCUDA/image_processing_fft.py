from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Read image file
img = Image.open('fox.jpg').convert('L')


# Convert image to numpy array
arr = np.array(img)


# FFT
arr2 = np.fft.fft2(arr)
arr3 = np.fft.fftshift(arr2)


#--------------------------------------------------------------------------
# Plot
#--------------------------------------------------------------------------
plt.ion()
fig = plt.figure(figsize=(20,7))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.set_title('Original')
ax2.set_title('FFT')

im1 = ax1.imshow(arr, cmap=plt.cm.Greys_r)
im2 = ax2.imshow(np.log10(np.abs(arr3)), cmap=plt.cm.Greys_r)

plt.tight_layout(pad=1)
plt.show(True)
