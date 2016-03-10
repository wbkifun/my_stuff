#------------------------------------------------------------------------------
# filename  : array_variable.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.2.23    start
#             2015.8.22    rename ArrayLike to ArrayAs
#
#
# description:
#   Generalized array variable for various langauge 
#   as Fortran, C, CUDA-C and OpenCL-C
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np




class Array(object):
    def __init__(self, platform, size, **kwargs):
        self.platform = platform
        self.size = size
        self.dtype = dtype = kwargs['dtype']
        self.name = name = kwargs['name']
        self.unit = unit = kwargs['unit']                    # MKS
        self.desc = desc = kwargs['desc']
        self.valid_range = valid_range = kwargs['valid_range']


        if platform.code_type == 'f90':
            self.data = np.zeros(size, dtype, order='F')
        else:
            self.data = np.zeros(size, dtype)


        if platform.code_type == 'cu':
            cuda = platform.cuda
            self.data_cu = cuda.mem_alloc_like(self.data)

        elif platform.code_type == 'cl':
            cl = platform.cl
            ctx = platform.context
            mf = cl.mem_flags
            
            if platform.device_type == 'CPU':
                self.data_cl = cl.Buffer(ctx, mf.ALLOC_HOST_PTR, self.data.nbytes)
            else:
                self.data_cl = cl.Buffer(ctx, self.data.nbytes)



    def set(self, input_data):
        assert self.data.size == input_data.size, 'Error: size mismatch. target:%s, input:%s'%(self.data.size, input_data.size)

        self.data[:] = input_data

        if self.platform.code_type == 'cu':
            cuda = self.platform.cuda
            cuda.memcpy_htod(self.data_cu, self.data)

        elif self.platform.code_type == 'cl':
            cl = self.platform.cl
            queue = self.platform.queue
            cl.enqueue_copy(queue, self.data_cl, self.data)


    
    def get(self):
        if self.platform.code_type == 'cu':
            cuda = self.platform.cuda
            cuda.memcpy_dtoh(self.data, self.data_cu)

        elif self.platform.code_type == 'cl':
            cl = self.platform.cl
            queue = self.platform.queue
            cl.enqueue_copy(queue, self.data, self.data_cl)

        return self.data




class ArrayAs(Array):
    def __init__(self, platform, arr, **kwargs):
        super(ArrayAs, self).__init__(platform, arr.size, \
                dtype=arr.dtype, name=kwargs['name'], unit=kwargs['unit'], \
                desc=kwargs['desc'], valid_range=kwargs['valid_range'])

        self.set(arr)
