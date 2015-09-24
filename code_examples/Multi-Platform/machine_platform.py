#------------------------------------------------------------------------------
# filename  : machine_platform.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.3.18    start
#             2015.3.23    append Function class
#             2015.9.23    modify the case of CPU-C with f2py
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


        if ctype == 'cu':
            import atexit
            import pycuda.driver as cuda

            cuda.init()
            dev = cuda.Device(device_number)
            ctx = dev.make_context()
            atexit.register(ctx.pop)

            self.cuda = cuda


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




    def source_compile(self, src, pyf):
        # The signature file(*.pyf) is only used for f90 and C

        if self.code_type == 'f90':
            from source_module import get_module_f90
            return get_module_f90(src, pyf)


        elif self.code_type == 'c':
            from source_module import get_module_c
            return get_module_c(src, pyf)


        elif self.code_type == 'cu':
            from pycuda.compiler import SourceModule
            return SourceModule(src)


        elif self.code_type == 'cl':
            return self.cl.Program(self.context, src).build()



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



    def prepare(self, arg_types, *args, **kwargs):
        '''
        arg_types :
            i: np.int32
            d: np.float64
            O: np.float64 array

        gsize : global thread size for CUDA and OpenCL
        '''

        ctype = self.platform.code_type
        if ctype in ['cu','cl']:
            if kwargs.has_key('gsize'):
                self.gsize = kwargs['gsize']
            else:
                if arg_types[0] == 'i':
                    self.gsize = args[0]
                else:
                    raise Exception, "When the code_type is 'cu' or 'cl' and the global size is not same with the first argument(integer), the gsize must be specified."


        self.preset_args = list()
<<<<<<< HEAD
        self.run_atypes = list()
=======
>>>>>>> 14aed050ee24fbb934191241ea3582bf78d8298b
        for atype, arg in zip(arg_types, args):
            if atype == 'i':
                self.preset_args.append( np.int32(arg) )

            elif atype == 'd':
                self.preset_args.append( np.float64(arg) )

            elif atype == 'O':
                if ctype in ['f90','c']:
                    self.preset_args.append( arg.data )

                elif ctype == 'cu':
                    self.preset_args.append( arg.data_cu )

                elif ctype == 'cl':
                    self.preset_args.append( arg.data_cl )
<<<<<<< HEAD

            elif atype in ['I','D']:
                # A capital letter means a argument given at calling.
                self.run_args.append(atype)

=======
>>>>>>> 14aed050ee24fbb934191241ea3582bf78d8298b

            else:
                assert False, "The arg_type '%s' is undefined."%(atype)


        if ctype == 'cu':
            self.block=(512,1,1)
            self.grid=(self.gsize//512+1,1)



    def prepared_call(self, *args):
        ctype = self.platform.code_type
        func = self.func
<<<<<<< HEAD
        run_args = self.preset_args
        run_atypes = self.run_atypes
        
        assert len(run_atypes) == len(args), 'len(run_atypes)=%d is not same as len(args)=%d'%(len(run_atypes),len(args))

        for atype, arg in zip(run_atypes, args):
            if atype == 'I':
                run_args.append( np.int32(arg) )

            elif atype == 'D':
                run_args.append( np.float64(arg) )

=======

        args = self.preset_args + list(args)
>>>>>>> 14aed050ee24fbb934191241ea3582bf78d8298b

        if ctype in ['f90', 'c']:
            func(*run_args)

        elif ctype == 'cu':
            func(*run_args, block=self.block, grid=self.grid)

        elif ctype == 'cl':
            func(self.platform.queue, (self.gsize,), None, *run_args)
