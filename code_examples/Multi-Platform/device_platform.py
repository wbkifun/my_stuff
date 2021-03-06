#------------------------------------------------------------------------------
# filename  : device_platform.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.10.29   split from machine_platform.py
#             2015.11.4    rename device.py -> device_platform.py
#             2016.3.10    representative function for DeviceLanguage classes
#                          insert Array and ArrayAs as members of the class
#             2016.3.11    add options; compiler name, compile option
#             2016.5.25    add load_module()
#             2016.5.26    add copy_array()
#             2016.9.8     add build_modules(), clean_modules()
#             2016.10.14   add thread_per_block argument for NVIDIA_GPU_CUDA()
#                          add work_group_size argument for OpenCL_Environemt()
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

import numpy as np
import os

import sys
from os.path import abspath, dirname, basename, join, exists
current_dpath = dirname(abspath(__file__))
sys.path.append(current_dpath)

from log import logger
from build import check_and_make_parameter_header, check_and_build, clean
from source_module import compile_using_f2py, get_module_from_file
from function import Function_F90_C, Function_CUDA, Function_OpenCL
import array_variable




def DevicePlatform(device, language, **kwargs):
    '''
    Return a DeviceLanguage class
    '''
    device = device.upper()
    language = language.upper()

    assert device in ['CPU', 'NVIDIA_GPU', 'INTEL_MIC'], "The device '%s' is not supported yet. Supported devices are 'CPU', 'NVIDIA_GPU' and 'INTEL_MIC'"%(device)

    supported_languages = { \
            'CPU'       :['F90', 'C', 'OPENCL'], 
            'NVIDIA_GPU':['CUDA', 'OPENCL'], 
            'INTEL_MIC' :['OPENCL'] } 

    assert language in supported_languages[device], "The language '%s' with the device '%s' is not supported. Supported languages with the device '%s' are %s"%(language, device, device, supported_languages[device])


    return globals()['%s_%s'%(device,language)](**kwargs)




class Environment:
    def __init__(self):
        pass


    def Array(self, shape, dtype=np.float64, name='', unit='', desc='', valid_range=None):
        return array_variable.Array(self, shape, \
                dtype=dtype, name=name, unit=unit, desc=desc, \
                valid_range=valid_range)


    def ArrayAs(self, arr, name='', unit='', desc='', valid_range=None):
        return array_variable.ArrayAs(self, arr, \
                name=name, unit=unit, desc=desc, valid_range=valid_range)


    def build_modules(self, base_dpath, code_type, generate_header, **kwargs):
        if generate_header:
            check_and_make_parameter_header(code_type, base_dpath)

        check_and_build(code_type, base_dpath, **kwargs)

        src_dir = {'f90':'f90', 'c':'c', 'cu':'cuda', 'cl':'opencl'}[code_type]
        build_dpath = join(base_dpath, src_dir, 'build')
        return build_dpath


    def clean_modules(self, base_dpath, code_type):
        clean(code_type, base_dpath)




class CPU_OpenMP_Environment(Environment):
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


    def copy_array(self, dst_array, src_array):
        dst_array.data[:] = src_array.data




class OpenCL_Environment(Environment):
    def __init__(self, vendor_name, device_type, device_number, work_group_size):
        self.cl_vendor_name = vendor_name
        self.cl_device_type = device_type
        self.device_number = device_number
        self.work_group_size = work_group_size

        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            platform = [p for p in platforms if vendor_name in p.vendor][0]
            all_devices = platform.get_devices()
            devices = [d for d in all_devices if cl.device_type.to_string(d.type)==device_type]
            context = cl.Context(devices)

        except Exception as e:
            logger.error("Error: OpenCL initialization error", exc_info=True)
            raise SystemExit


        max_devices = len(devices)
        if max_devices == 0:
            logger.error("Error: There is no OpenCL platform which has device_type as %s."%device_type)
            raise SystemExit

        elif device_number >= max_devices:
            logger.error("Error: The given device_number(%d) is bigger than physical devices(%d)."%(device_number, max_devices))
            raise SystemExit

        else:
            queue = cl.CommandQueue(context, devices[device_number])

            self.cl = cl
            self.devices = devices
            self.context = context
            self.queue = queue

            # Show more compiler message
            os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

            # Prevent a warning message when a Program.build() is called.
            os.environ['PYOPENCL_NO_CACHE'] = '1'


    def copy_array(self, dst_array, src_array):
        self.cl.enqueue_copy(self.queue, dst_array.data_cl, src_array.data_cl)


    def source_compile(self, src):
        prg = self.cl.Program(self.context, src)
        lib = prg.build()

        return lib


    def build_modules(self, base_dpath, generate_header):
        return super(OpenCL_Environment, self).build_modules( \
                base_dpath, self.code_type, generate_header, \
                opencl_vendor_name=self.cl_vendor_name, \
                opencl_device_type=self.cl_device_type)


    def clean_modules(self, base_dpath):
        return super(OpenCL_Environment, self).clean_modules(base_dpath, self.code_type)


    def load_module(self, build_dpath, module_name):
        clbin_fpath = join(build_dpath, module_name+'.clbin')
        with open(clbin_fpath, 'rb') as f:
            binary = f.read()
            binaries = [binary for d in self.devices]
            prg = self.cl.Program(self.context, self.devices, binaries)
            lib = prg.build()

        return lib


    def get_function(self, lib, func_name):
        func = getattr(lib, func_name)

        return Function_OpenCL(self.queue, func, self.work_group_size)




class CPU_F90(CPU_OpenMP_Environment):
    def __init__(self, use_cpu_cores='all', **kwargs):
        super(CPU_F90, self).__init__(use_cpu_cores)
        self.device_type = 'cpu'
        self.language    = 'f90'
        self.code_type   = 'f90'


    def source_compile(self, src, compiler='gnu', flags='', opt_flags='-O3'):
        # Compile directly for a source code as string.
        # A signature file(*.pyf) is generated automatically.

        logger.debug('source code: {}'.format(src))

        import tempfile
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.f90', delete=True)
        tmpfile.file.write(src)
        tmpfile.file.flush()

        compile_using_f2py(tmpfile.name, compiler, flags, opt_flags)

        mod_name = basename(tmpfile.name).split('.')[0]
        return get_module_from_file('/tmp/build/', mod_name, self.code_type)


    def build_modules(self, base_dpath, generate_header):
        return super(CPU_F90, self).build_modules(base_dpath, self.code_type, generate_header)


    def clean_modules(self, base_dpath):
        return super(CPU_F90, self).clean_modules(base_dpath, self.code_type)


    def load_module(self, build_dpath, module_name):
        return get_module_from_file(build_dpath, module_name, self.code_type)


    def get_function(self, lib, func_name, **kwargs):
        if 'f90_mod_name' in kwargs.keys():
            f90_mod = getattr(lib, kwargs['f90_mod_name'])
            print('f90_mod', f90_mod)
            func = getattr(f90_mod, func_name)
            print('func', func)

        else:
            func = getattr(lib, func_name)

        return Function_F90_C(func)




class CPU_C(CPU_OpenMP_Environment):
    def __init__(self, use_cpu_cores='all', **kwargs):
        super(CPU_C, self).__init__(use_cpu_cores)
        self.device_type = 'cpu'
        self.language    = 'c'
        self.code_type   = 'c'


    def source_compile(self, src, compiler='gnu', flags='', opt_flags='-O3'):
        # Compile directly for a source code as string.
        # A signature file(*.pyf) is generated automatically.
        # Fortran-like intents should be defined.

        logger.debug('source code: {}'.format(src))

        import tempfile
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=True)
        tmpfile.file.write(src)
        tmpfile.file.flush()

        compile_using_f2py(tmpfile.name, compiler, flags, opt_flags)

        mod_name = basename(tmpfile.name).split('.')[0]
        return get_module_from_file('/tmp/build/', mod_name, self.code_type)


    def build_modules(self, base_dpath, generate_header):
        return super(CPU_C, self).build_modules(base_dpath, self.code_type, generate_header)


    def clean_modules(self, base_dpath):
        return super(CPU_C, self).clean_modules(base_dpath, self.code_type)


    def load_module(self, build_dpath, module_name):
        return get_module_from_file(build_dpath, module_name, self.code_type)


    def get_function(self, lib, func_name, **kwargs):
        func = getattr(lib, func_name)

        return Function_F90_C(func)




class CPU_OPENCL(OpenCL_Environment):
    def __init__(self, vendor_name='Intel', work_group_size=None, **kwargs):
        super(CPU_OPENCL, self).__init__(vendor_name, 'CPU', 0, work_group_size)
        self.device_type = 'cpu'
        self.language    = 'opencl'
        self.code_type   = 'cl'




class INTEL_MIC_OPENCL(OpenCL_Environment):
    def __init__(self, device_number=0, work_group_size=None, **kwargs):
        super(INTEL_MIC_OPENCL, self).__init__('Intel', 'ACCELERATOR', device_number, work_group_size)
        self.device_type = 'intel_mic'
        self.language    = 'opencl'
        self.code_type   = 'cl'




class NVIDIA_GPU_OPENCL(OpenCL_Environment):
    def __init__(self, device_number=0, work_group_size=None, **kwargs):
        super(NVIDIA_GPU_OPENCL, self).__init__('NVIDIA', 'GPU', device_number, work_group_size)
        self.device_type = 'nvidia_gpu'
        self.language    = 'opencl'
        self.code_type   = 'cl'




class NVIDIA_GPU_CUDA(Environment):
    def __init__(self, device_number=0, thread_per_block=512, **kwargs):
        self.device_number = device_number
        self.thread_per_block = thread_per_block
        self.device_type = 'nvidia_gpu'
        self.language    = 'cuda'
        self.code_type   = 'cu'

        try:
            import pycuda.driver as cuda
            cuda.init()

        except Exception as e:
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


    def copy_array(self, dst_array, src_array):
        self.cuda.memcpy_dtod(dst_array.data_cu, src_array.data_cu, dst_array.nbytes)


    def source_compile(self, src):
        from pycuda.compiler import SourceModule
        lib = SourceModule(src)

        return lib


    def build_modules(self, base_dpath, generate_header):
        return super(NVIDIA_GPU_CUDA, self).build_modules(base_dpath, self.code_type, generate_header)


    def clean_modules(self, base_dpath):
        return super(NVIDIA_GPU_CUDA, self).clean_modules(base_dpath, self.code_type)


    def load_module(self, build_dpath, module_name):
        cubin_fpath = join(build_dpath, module_name+'.cubin')
        assert exists(cubin_fpath), "Error: '{}' is not found.".format(cubin_fpath)
        return self.cuda.module_from_file(cubin_fpath)

        
    def get_function(self, lib, func_name):
        func = lib.get_function(func_name)

        return Function_CUDA(func, self.thread_per_block)


    def set_constant(self, lib, var_name, var):
        device_ptr, byte_size = lib.get_global(var_name)
        self.cuda.memcpy_htod(device_ptr, var.data)
