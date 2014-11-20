from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import pycuda.driver as cuda
from pyfft.cuda import Plan



def pad_power_of_two(arr):
    widths = [0 for n in arr.shape]

    for axis, n in enumerate(arr.shape):
        for i in xrange(100):
            if 2**i > n:
                widths[axis] = 2**i - n
                break
    pad_arr = np.pad(arr, [(0,width) for width in widths], 'constant', \
            constant_values=(0,0))

    return pad_arr



# Read image file
img = Image.open('fox.jpg').convert('L')


# Convert image to numpy array
arr = np.array(img, dtype=np.float32)
pad_arr = pad_power_of_two(arr)
pad_arr2 = np.empty_like(pad_arr)


# FFT with CUDA
cuda.init()
ctx = cuda.Device(0).make_context()
strm = cuda.Stream()

pad_arr_gpu = cuda.to_device(pad_arr)
plan = Plan(pad_arr.shape, dtype=np.float32, context=ctx, stream=strm)
plan.execute(pad_arr_gpu)
cuda.memcpy_dtoh(pad_arr2, pad_arr_gpu)
pad_arr3 = np.fft.fftshift(pad_arr2)


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
