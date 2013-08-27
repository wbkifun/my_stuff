from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule



#--------------------------------------------------------------------------
# compile and load the cuda kernel
#--------------------------------------------------------------------------
kernels='''
__global__ void image_process(int nx, int ny, float *arr) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int pix = tid*3;

    if (tid < nx*ny) {
        // RGB exchange
        arr[pix+0] = arr[pix+1];   // R <- G
        arr[pix+1] = arr[pix+2];   // G <- B
        arr[pix+2] = arr[pix+0];   // B <- R
    }
}
'''
mod = SourceModule(kernels)
image_process = mod.get_function('image_process')



#--------------------------------------------------------------------------
# array create from a image file(png)
#--------------------------------------------------------------------------
# read image file
img = Image.open("two_girls.png")

# convert image to numpy array
arr = np.array(img, dtype=np.float32)
print 'shape:', arr.shape
print 'dtype:', arr.dtype
#print arr

# copy the numpy array to the cuda array
arr_gpu = cuda.to_device(arr)



#--------------------------------------------------------------------------
# kernel call
#--------------------------------------------------------------------------
nx, ny = arr.shape[:-1]
tpb = 256               # thread/block
bpg = nx*ny//tpb + 1    # block/grid
image_process(np.int32(nx), np.int32(ny), arr_gpu, block=(tpb,1,1), grid=(bpg,1))
cuda.memcpy_dtoh(arr, arr_gpu)



#--------------------------------------------------------------------------
# plot
#--------------------------------------------------------------------------
plt.imshow(arr)
plt.show()
