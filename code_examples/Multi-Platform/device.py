#------------------------------------------------------------------------------
# filename  : device.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.10.29   split from machine_platform.py
#
#
# description:
#   Generalized interface to call functions from various code languages
#
# support processor-language pairs:
#   CPU        - Fortran 90/95, C, OpenCL
#   NVIDIA GPU - CUDA
#   AMD GPU    - OpenCL
#   Intel MIC  - OpenCL
#   FPGA       - OpenCL
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import os

from util.log import logger
from function import Function_F90_C, Function_CUDA, Function_OpenCL




class CUDA_Environment(object):
    def __init__(self, gpu_number):
        self.gpu_number = gpu_number

        try:
            import pycuda.driver as cuda

            cuda.init()
            device = cuda.Device(gpu_number)
            context = device.make_context()
        except Exception, e:
            logger.error("Error: CUDA initialization error", exc_info=True)
            raise SystemExit

        import atexit
        atexit.register(context.pop)

        self.cuda = cuda
        self.device = device
        self.context = context




class OpenCL_Environment(object):
    def __init__(self, platform_number, device_number):
        self.platform_number = platform_number
        self.device_number = device_number

        try:
            import pyopencl as cl

            platforms = cl.get_platforms()
            devices = platforms[platform_number].get_devices()
            context = cl.Context(devices)
            queue = cl.CommandQueue(context, devices[device_number])

        except Exception, e:
            logger.error("Error: OpenCL initialization error", exc_info=True)
            raise SystemExit

        self.cl = cl
        self.context = context
        self.queue = queue




class CPU_F90(object):
    def __init__(self):
        self.machine_type = 'CPU'
        self.code_type = 'f90'


    def startup(self):
        pass


    def source_compile(self, src):
        # The f2py is used for compiling *.f90 or *.c codes.
        # The f2py requires a signature file(*.pyf).
        # Although the signature file is automatically generated from *.f90,
        # the integer argument for dimension is automatically skipped.
        # So we generate the signature file explicitly.

        from source_module import make_signature_f90, get_module_f90
        pyf = make_signature_f90(src)
        logger.debug('source code:\n%s\n'%src)
        logger.debug('signature for f2py:\n%s\n'%pyf)
        lib = get_module_f90(src, pyf)

        return lib


    def get_function(self, lib, func_name, **kwargs):
        if kwargs.has_key('f90_mod_name'):
            f90_mod = getattr(lib, kwargs['f90_mod_name'])
            func = getattr(f90_mod, func_name)

        else:
            func = getattr(lib, func_name)

        return Function_F90_C(func)




class CPU_C(object):
    def __init__(self):
        self.machine_type = 'CPU'
        self.code_type = 'c'


    def startup(self):
        pass


    def source_compile(self, src):
        from source_module import make_signature_c, get_module_c

        pyf = make_signature_c(src)
        logger.debug('source code:\n%s\n'%src)
        logger.debug('signature for f2py:\n%s\n'%pyf)
        lib = get_module_c(src, pyf)

        return lib


    def get_function(self, lib, func_name, **kwargs):
        func = getattr(lib, func_name)

        return Function_F90_C(func)




class CPU_OpenCL(OpenCL_Environment):
    def __init__(self, platform_number, device_number):
        self.platform_number = platform_number
        self.device_number = device_number
        self.machine_type = 'CPU'
        self.code_type = 'cl'


    def startup(self):
        super(CPU_OpenCL, self).__init__( \
                self.platform_number, self.device_number)


    def source_compile(self, src):
        #os.environ['PYOPENCL_NO_CACHE'] = '1'
        lib = self.cl.Program(self.context, src).build()

        return lib


    def get_function(self, lib, func_name, **kwargs):
        func = getattr(lib, func_name)

        return Function_OpenCL(self.queue, func)




class NVIDIA_GPU_CUDA(CUDA_Environment):
    def __init__(self, device_number):
        self.device_number = device_number
        self.machine_type = 'NVIDIA_GPU'
        self.code_type = 'cu'


    def startup(self):
        super(NVIDIA_GPU_CUDA, self).__init__(self.device_number)


    def source_compile(self, src):
        from pycuda.compiler import SourceModule
        lib = SourceModule(src)

        return lib


    def get_function(self, lib, func_name, **kwargs):
        func = lib.get_function(func_name)

        return Function_CUDA(func)
