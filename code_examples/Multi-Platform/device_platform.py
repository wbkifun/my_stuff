#------------------------------------------------------------------------------
# filename  : device_platform.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.10.29   split from machine_platform.py
#             2015.11.4    rename device.py -> device_platform.py
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




class CPU_OpenMP_Environment(object):
    def __init__(self, use_cpu_cores):
        self.use_cpu_cores = use_cpu_cores

        import multiprocessing as mulp
        max_cores = mulp.cpu_count()

        if use_cpu_cores == 'all':
            use_cpu_cores = max_cores

        elif use_cpu_cores > max_cores:
            logger.error("Error: The given use_cpu_cores(%d) is bigger than physical CPU cores(%d)."%(use_cpu_cores, max_cores))
            raise SystemExit

        os.environ['OMP_NUM_THREADS'] = '%d'%(use_cpu_cores)




class CUDA_Environment(object):
    def __init__(self, device_number):
        self.device_number = device_number

        try:
            import pycuda.driver as cuda
            cuda.init()

        except Exception, e:
            logger.error("Error: CUDA initialization error", exc_info=True)
            raise SystemExit

        max_devices = cuda.Device.count()
        if max_devices == 0:
            logger.error("Error: There is no CUDA device (NVIDIA GPU).")
            raise SystemExit

        elif device_number >= max_devices:
            logger.error("Error: The given device_number(%d) is bigger than physical GPU devices(%d)."%(device_number, max_devices))
            raise SystemExit

        else:
            device = cuda.Device(device_number)
            context = device.make_context()

            import atexit
            atexit.register(context.pop)

            self.cuda = cuda
            self.device = device
            self.context = context




class OpenCL_Environment(object):
    def __init__(self, device_type, device_number):
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()

        except Exception, e:
            logger.error("Error: OpenCL initialization error", exc_info=True)
            raise SystemExit

        for platform_number, platform in enumerate(platforms):
            dev_type = getattr(cl.device_type,device_type.upper())
            devices = platform.get_devices(dev_type)
            if len(devices) > 0: break

        max_devices = len(devices)
        if max_devices == 0:
            logger.error("Error: There is no OpenCL platform which has device_type as %s."%device_type)
            raise SystemExit

        elif device_number >= max_devices:
            logger.error("Error: The given device_number(%d) is bigger than physical GPU devices(%d)."%(device_number, max_devices))
            raise SystemExit

        else:
            devices = platforms[platform_number].get_devices()
            context = cl.Context(devices)
            queue = cl.CommandQueue(context, devices[device_number])

            self.cl = cl
            self.context = context
            self.queue = queue

            # Prevent a warning message when a Program.build() is called.
            #os.environ['PYOPENCL_NO_CACHE'] = '1'




class CPU_F90(CPU_OpenMP_Environment):
    def __init__(self, use_cpu_cores='all'):
        super(CPU_F90, self).__init__(use_cpu_cores)
        self.device_type = 'CPU'
        self.code_type = 'f90'


    def source_compile(self, src):
        # The f2py is used for compiling *.f90 or *.c codes.
        # The f2py requires a signature file(*.pyf).
        # Although the signature file is automatically generated from *.f90,
        # the integer argument for dimension is automatically skipped.
        # So we generate the signature file explicitly.

        from source_module import make_signature_f90, get_module_f90
        pyf = make_signature_f90(src)
        logger.debug('source code:%s'%src)
        logger.debug('signature for f2py:%s'%pyf)
        lib = get_module_f90(src, pyf)

        return lib


    def get_function(self, lib, func_name, **kwargs):
        if kwargs.has_key('f90_mod_name'):
            f90_mod = getattr(lib, kwargs['f90_mod_name'])
            func = getattr(f90_mod, func_name)

        else:
            func = getattr(lib, func_name)

        return Function_F90_C(func)




class CPU_C(CPU_OpenMP_Environment):
    def __init__(self, use_cpu_cores='all'):
        super(CPU_C, self).__init__(use_cpu_cores)
        self.device_type = 'CPU'
        self.code_type = 'c'


    def source_compile(self, src):
        from source_module import make_signature_c, get_module_c

        pyf = make_signature_c(src)
        logger.debug('source code:%s'%src)
        logger.debug('signature for f2py:%s'%pyf)
        lib = get_module_c(src, pyf)

        return lib


    def get_function(self, lib, func_name, **kwargs):
        func = getattr(lib, func_name)

        return Function_F90_C(func)




class CPU_OpenCL(OpenCL_Environment):
    def __init__(self):
        super(CPU_OpenCL, self).__init__('CPU', device_number=0)
        self.device_type = 'CPU'
        self.code_type = 'cl'


    def source_compile(self, src):
        lib = self.cl.Program(self.context, src).build()

        return lib


    def get_function(self, lib, func_name, **kwargs):
        func = getattr(lib, func_name)

        return Function_OpenCL(self.queue, func)




class NVIDIA_GPU_CUDA(CUDA_Environment):
    def __init__(self, device_number):
        super(NVIDIA_GPU_CUDA, self).__init__(device_number)
        self.device_type = 'NVIDIA_GPU'
        self.code_type = 'cu'


    def source_compile(self, src):
        from pycuda.compiler import SourceModule
        lib = SourceModule(src)

        return lib


    def get_function(self, lib, func_name, **kwargs):
        func = lib.get_function(func_name)

        return Function_CUDA(func)