#------------------------------------------------------------------------------
# filename  : array_variable.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.2.23    start
#
#
# description:
#   Generalized array variable for various langauge 
#   as Fortran, C, CUDA-C and OpenCL-C
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np




class Array(object):
    def __init__(self, platform, shape, dtype='f8', fullname='', unit='', valid_range=None):
        self.platform = platform
        self.shape = shape
        self.dtype = dtype
        self.fullname = fullname
        self.unit = unit                        # MKS
        self.valid_range = valid_range


        if platform.code_type == 'f90':
            self.data = np.zeros(shape, dtype, order='F')
        else:
            self.data = np.zeros(shape, dtype)


        if platform.code_type == 'cu':
            cuda = platform.cuda
            self.data_cu = cuda.mem_alloc_like(self.data)

        elif platform.code_type == 'cl':
            cl = platform.cl
            ctx = platform.context
            mf = cl.mem_flags
            
            if platform.machine_type == 'cpu':
                self.data_cl = cl.Buffer(ctx, mf.ALLOC_HOST_PTR, self.data.nbytes)
            else:
                self.data_cl = cl.Buffer(ctx, self.data.nbytes)



    def set_data(self, input_data):
        assert self.data.shape == input_data.shape, 'Error: shape mismatch. target:%s, input:%s'%(self.data.shape, input_data.shape)

        self.data[:] = input_data

        if self.platform.code_type == 'cu':
            cuda = self.platform.cuda
            cuda.memcpy_htod(self.data_cu, self.data)

        elif self.platform.code_type == 'cl':
            cl = self.platform.cl
            queue = self.platform.queue
            cl.enqueue_copy(queue, self.data_cl, self.data)


    
    def get_data(self):
        if self.platform.code_type == 'cu':
            cuda = self.platform.cuda
            cuda.memcpy_dtoh(self.data, self.data_cu)

        elif self.platform.code_type == 'cl':
            cl = self.platform.cl
            queue = self.platform.queue
            cl.enqueue_copy(queue, self.data, self.data_cl)

        return self.data




class ArrayLike(Array):
    def __init__(self, platform, arr, fullname='', unit='', valid_range=None):
        super(ArrayLike, self).__init__(platform, arr.shape, arr.dtype, fullname, unit, valid_range)

        self.set_data(arr)
