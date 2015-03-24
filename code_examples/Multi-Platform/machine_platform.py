#------------------------------------------------------------------------------
# filename  : machine_platform.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.3.18    start
#             2015.3.23    append Function class
#
#
# description:
#   Interface to call functions for various machine platforms
#
# support processors:
#   Intel CPU, AMD CPU, NVIDIA GPU, AMD GPU, Intel MIC
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np




class MachinePlatform(object):
    def __init__(self, machine_type, code_type, device_number=0, print_on=True):
        self.machine_type = mtype = machine_type.lower()
        self.code_type = ctype = code_type.lower()

        support_types = {'cpu':['f90','c','cl'], \
                         'nvidia gpu':['cu'], \
                         'amd gpu':['cl'], \
                         'intel mic':['cl']}

        ctype_fullname = {'f90':'Fortran 90/95', \
                          'c':'C', \
                          'cu':'CUDA-C', \
                          'cl':'OpenCL-C'}

        if print_on:
            print 'Machine type : %s' % (mtype.upper())
            print 'Code type    : %s (%s)' % (ctype, ctype_fullname[ctype])
            print 'Device number : %d\n' % (device_number)

        assert mtype in support_types.keys(), "The support machine_type is one of the %s, machine_type=%s"%([t.upper() for t in support_types.keys()], machine_type)

        assert ctype in support_types[mtype], "The machine_type %s only supports one of the code_type %s. code_type=%s" % (mtype.upper(), support_types[mtype], ctype)


        if ctype == 'f90':
            from source_module import get_module_f90
            self.source_compile = get_module_f90


        elif ctype == 'c':
            from source_module import get_module_c
            self.source_compile = get_module_c


        elif ctype == 'cu':
            import atexit
            import pycuda.driver as cuda
            from pycuda.compiler import SourceModule

            cuda.init()
            dev = cuda.Device(device_number)
            ctx = dev.make_context()
            atexit.register(ctx.pop)

            self.cuda = cuda
            self.source_compile = SourceModule


        elif ctype == 'cl':
            import pyopencl as cl

            platforms = cl.get_platforms()

            if len(platforms) == 1:
                platform_number = 0

            else:
                print '%d platforms are founded.' % (len(platforms))
                for i, platform in enumerate(platforms):
                    device = platform.get_devices()[0]
                    print '\t%d: %s' % (i, device.name)

                while True:
                    platform_number = input('Select platform: ')
                    if platform_number in range(len(platforms)):
                        break
                    else:
                        print 'Wrong platform number.'

            devices = platforms[platform_number].get_devices()
            context = cl.Context(devices)
            queue = cl.CommandQueue(context, devices[device_number])

            self.cl = cl
            self.context = context
            self.queue = queue
            self.source_compile = lambda src : cl.Program(context, src).build()



    def get_function(self, lib, func_name, **kwargs):
        self.func_name = func_name

        if kwargs.has_key('f90_mod_name') and self.code_type == 'f90':
            f90_mod = getattr(lib, kwargs['f90_mod_name'])
            func = getattr(f90_mod, func_name)

        elif self.code_type == 'cu':
            func = lib.get_function(func_name)

        else:
            func = getattr(lib, func_name)

        return Function(self, func)




class Function(object):
    def __init__(self, platform, func):
        self.platform = platform
        self.func = func



    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)



    def prepare(self, arg_types, *raw_args, **kwargs):
        '''
        arg_types :
            i: np.int32
            d: np.float64
            O: np.float64 array
            I: np.int32 (optional)

        gsize : global thread size for CUDA and OpenCL
        '''

        ctype = self.platform.code_type
        if ctype in ['cu','cl']:
            assert kwargs.has_key('gsize'), "When the code_type is 'cu' or 'cl', the gsize must be specified."
            self.gsize = kwargs['gsize']

        self.args = list()
        for atype, arg in zip(arg_types, raw_args):
            if atype == 'i':
                self.args.append( np.int32(arg) )

            elif atype == 'd':
                self.args.append( np.float64(arg) )

            elif atype == 'O':
                if ctype in ['f90','c']:
                    self.args.append( arg.data )

                elif ctype == 'cu':
                    self.args.append( arg.data_cu )

                elif ctype == 'cl':
                    self.args.append( arg.data_cl )

            elif atype == 'I':
                if ctype in ['cu', 'cl']:
                    self.args.append( np.int32(arg) )

            else:
                assert False, "The arg_type '%s' is undefined."%(atype)



    def prepared_call(self):
        ctype = self.platform.code_type
        func = self.func
        args = self.args

        if ctype in ['f90', 'c']:
            func(*args)

        elif ctype == 'cu':
            func(*args, block=(256,1,1), grid=(self.gsize//256+1,1))

        elif ctype == 'cl':
            func(self.platform.queue, (self.gsize,), None, *args)
