#------------------------------------------------------------------------------
# filename  : machine_platform.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.3.18    start
#             2015.3.23    append Function class
#             2015.9.23    modify the case of CPU-C with f2py
#             2015.9.24    modify for more flexible arguments
#             2015.10.30   extend to heterogeneous platform with multiprocessing
#
#
# description:
#   Interface to call functions for various machine platforms
#   Operate with multi-devices (e.g. GPU+GPU+CPU)
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
import multiprocessing as mulp
import os

from util.log import logger




class DeviceProcess(mulp.Process):
    def __init__(self):
        #Process.__init__(self)
        super(DeviceQueue, self).__init__(self)
        self.task_queue = mulp.JoinableQueue()


    def run(self):
        while True:
            if self.task_queue.get() == 'exit':
                self.task_queue.task_done()
                break

            else:
                func, args, wait_list, event = self.task_queue.get()

                if event: event.clear()  # set False from mulp.Event() 

                # wait for prerequisite conditions
                for evt in wait_list: evt.wait()

                func(*args)

                if event: event.set()    # set True


    def finalize(self):
        self.task_queue.put('exit')
        self.task_queue.join()
                



class MachinePlatform(object):
    def __init__(self, device_list):
        self.device_list = device_list  # (device_type, code_type, num_devices)

        self.device_platforms = self.create_device_platforms()
        self.device_processes = self.create_device_processes()

        self.machine_types = [dev.machine_type for dev in self.device_platforms]
        self.code_types = [dev.code_type for dev in self.device_platforms]



    def create_device_platforms(self):
        device_platforms = list()

        for device_type, code_type, num_devices in self.device_list:
            device_type = device_type.upper()
            code_type = code_type.lower()
            num_devices = num_devices if type(num_devices) == int else num_devices.lower()

            if device_type == 'CPU':
                max_cores = mulp.cpu_count()
                if num_devices == 'all': num_devices = max_cores

                if num_devices > max_cores:
                    logger.error("Error: The given CPU count(%d) is bigger than physical CPU cores(%d)."%(num_devices, max_cores))
                    raise SystemExit

                os.environ['OMP_NUM_THREADS'] = '%d'%(num_devices)


                if code_type == 'f90':
                    from device import CPU_F90
                    device_platforms.append( CPU_F90() )

                elif code_type == 'c':
                    from device import CPU_C
                    device_platforms.append( CPU_C() )

                elif code_type == 'cl':
                    from device import CPU_OpenCL
                    platform_number, num_cl_dev = self.find_opencl_device(device_type)
                    if num_cl_dev != 1:
                        logger.error("Error: The number of OpenCL CPU device is not 1. %d CPU devices are founded."%(num_cl_dev))
                        raise SystemExit

                    device_platforms.append( CPU_OpenCL(platform_number, 0) )

                else:
                    logger.error("Error: The device_type '%s' does not support the code_type '%s'."%(device_type, code_type))
                    raise SystemExit


            elif device_type == 'NVIDIA_GPU':
                from device import NVIDIA_GPU_CUDA

                if code_type == 'cu':
                    num_cu_dev = find_cuda_device()
                    if num_devices > num_cu_dev:
                        logger.error("Error: The given NVIDIA GPU count(%d) is bigger than physical devices(%d)."%(num_devices,num_cu_dev))
                        raise SystemExit

                    if num_devices == 'all': num_devices = num_cu_dev
                    for i in xrange(num_devices):
                        device_platforms.append( NVIDIA_GPU_CUDA(i) )

                else:
                    logger.error("Error: The device_type '%s' does not support the code_type '%s'."%(device_type, code_type))
                    raise SystemExit


            else:
                logger.error("Error: The device_type '%s' is not supported yet."%(device_type))
                raise SystemExit


        return device_platforms



    def find_cuda_device(self):
        try:
            import pycuda.driver as cuda
            cuda.init()

        except Exception, e:
            logger.error("Error: OpenCL initialization error", exc_info=True)
            raise SystemExit
        
        num_devices = cuda.Device.count()
        if num_devices > 0:
            return num_devices
        else:
            logger.error("Error: There is no CUDA device (NVIDIA GPU).")
            raise SystemExit



    def find_opencl_device(self, device_type):
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()

        except Exception, e:
            logger.error("Error: OpenCL initialization error", exc_info=True)
            raise SystemExit
        

        for platform_number, platform in enumerate(platforms):
            dev_type = getattr(cl.device_type,device_type.upper())
            devices = platform.get_devices(dev_type)

            if len(devices) > 0:
                return platform_number, len(devices)

        logger.error("Error: There is no OpenCL platform which has device_type as %s."%device_type)
        raise SystemExit



    def create_device_processes(self):
        device_processes = list()

        for device_platform in self.device_platforms:
            device_process = DeviceProcess()
            device_process.start()
            device_process.task_queue.put((device_platform.startup, [], [], None))
            self.device_processes.append(device_process)

        return device_processes



    def finalize(self):
        for device_process in self.device_processes:
            device_process.finalize()



    def source_compile(self, src_list):
        lib_list = list()

        for device_platform, src in zip(self.device_platforms, src_list):
            lib = device_platform.source_compile(src)
            lib_list.append(lib)

        return lib_list



    def get_function(self, lib_list, func_name, **kwargs):
        func_list = list()

        for device_platform, lib in zip(self.device_platforms, lib_list):
            func = device_platform.get_function(lib, func_name, **kwargs)
            func_list.append(func)

        return func_list



    def func_prepare(self, func_list, arg_type, *args, **kwargs):
        for func in func_list:
            func.prepare(arg_type, *args, **kwargs)



    def func_prepared_call(self, func_list, *args):
        for func in func_list:
            func.prepared_call(*args)
