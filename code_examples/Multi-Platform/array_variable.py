#------------------------------------------------------------------------------
# filename  : array_variable.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.2.23    start
#             2015.8.22    rename ArrayLike to ArrayAs
#             2016.5.25    change the 'size' argument to 'shape'
#
#
# description:
#   Generalized array variable for various langauge 
#   as Fortran, C, CUDA-C and OpenCL-C
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np




class Array(object):
    def __init__(self, platform, shape, **kwargs):
        self.platform = platform
        self.shape = shape

        self.dtype = dtype = kwargs['dtype']
        self.name = name = kwargs['name']
        self.unit = unit = kwargs['unit']                    # MKS
        self.desc = desc = kwargs['desc']
        self.valid_range = valid_range = kwargs['valid_range']


        if platform.code_type == 'f90':
            self.data = np.zeros(shape, dtype, order='F')
        else:
            self.data = np.zeros(np.prod(shape), dtype)

        self.size = self.data.size
        self.nbytes = self.data.nbytes
        self.ndim = 1 if type(shape)==int else len(shape)


        if platform.code_type == 'cu':
            cuda = platform.cuda
            #self.data_cu = cuda.mem_alloc_like(self.data)
            self.data_cu = cuda.to_device(self.data)

        elif platform.code_type == 'cl':
            cl = platform.cl
            ctx = platform.context
            mf = cl.mem_flags
            
            if platform.device_type == 'cpu':
                self.data_cl = cl.Buffer(ctx, mf.ALLOC_HOST_PTR, self.data.nbytes)
            else:
                self.data_cl = cl.Buffer(ctx, self.data.nbytes)



    def sync_htod(self):
        if self.platform.code_type == 'cu':
            cuda = self.platform.cuda
            cuda.memcpy_htod(self.data_cu, self.data)

        elif self.platform.code_type == 'cl':
            cl = self.platform.cl
            queue = self.platform.queue
            cl.enqueue_copy(queue, self.data_cl, self.data, is_blocking=True)



    def sync_dtoh(self):
        if self.platform.code_type == 'cu':
            cuda = self.platform.cuda
            cuda.memcpy_dtoh(self.data, self.data_cu)

        elif self.platform.code_type == 'cl':
            cl = self.platform.cl
            queue = self.platform.queue
            cl.enqueue_copy(queue, self.data, self.data_cl, is_blocking=True)



    def set(self, input_data):
        assert self.data.size == input_data.size, 'Error: size mismatch. target:%s, input:%s'%(self.data.size, input_data.size)

        if self.platform.code_type == 'f90':
            self.data[:] = input_data.ravel().reshape(self.data.shape, order='F')
        else:
            self.data[:] = input_data.ravel()

        self.sync_htod()


    
    def get(self):
        self.sync_dtoh()
        return self.data




class ArrayAs(Array):
    def __init__(self, platform, arr, **kwargs):
        super(ArrayAs, self).__init__(platform, arr.shape, \
                dtype=arr.dtype, name=kwargs['name'], unit=kwargs['unit'], \
                desc=kwargs['desc'], valid_range=kwargs['valid_range'])

        self.set(arr)
