#------------------------------------------------------------------------------
# filename  : machine_platform.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.3.18    start
#             2015.3.23    append Function class
#             2015.9.23    modify the case of CPU-C with f2py
#             2015.9.24    modify for more flexible arguments
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
    def __init__(self, machine_type, code_type, device_number=0, print_on=False):
        self.machine_type = mtype = machine_type.lower()
        self.code_type = code_type = code_type.lower()

        support_types = {'cpu':['f90','c','cl'], \
                         'nvidia gpu':['cu'], \
                         'amd gpu':['cl'], \
                         'intel mic':['cl']}

        code_type_fullname = {'f90':'Fortran 90/95', \
                              'c':'C', \
                              'cu':'CUDA-C', \
                              'cl':'OpenCL-C'}

        if print_on:
            print 'Machine type : %s' % (mtype.upper())
            print 'Code type    : %s (%s)' % (code_type, code_type_fullname[code_type])
            print 'Device number : %d\n' % (device_number)

        assert mtype in support_types.keys(), "The support machine_type is one of the %s, machine_type=%s"%([t.upper() for t in support_types.keys()], machine_type)

        assert code_type in support_types[mtype], "The machine_type %s only supports one of the code_type %s. code_type=%s" % (mtype.upper(), support_types[mtype], code_type)


        if code_type == 'cu':
            import atexit
            import pycuda.driver as cuda

            cuda.init()
            dev = cuda.Device(device_number)
            ctx = dev.make_context()
            atexit.register(ctx.pop)

            self.cuda = cuda


        elif code_type == 'cl':
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
            pyf2 = pyf.replace('intent(c)', '! intent(c)')
            return get_module_f90(src, pyf2)


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
            i,I: np.int32
            d,D: np.float64
            o,O: numpy array

        gsize : global thread size for CUDA and OpenCL

        Lowercase letter means that it can be set before function call.
        Uppercase letter means that it should set when function call.
        '''
        code_type = self.platform.code_type
        code_types = ['f90', 'c', 'cu', 'cl']


        #--------------------------------------------------------
        # Global thread size in case of CUDA and OpenCL
        #--------------------------------------------------------
        if code_type in ['cu','cl']:
            if kwargs.has_key('gsize'):
                # explicit
                self.gsize = kwargs['gsize']

            else:
                # implicit
                if arg_types[0] == 'i':
                    self.gsize = args[0]
                else:
                    raise Exception, "When the code_type is 'cu' or 'cl' and the global size is not same with the first argument(integer), the gsize must be specified."

        if code_type == 'cu':
            self.block=(512,1,1)
            self.grid=(self.gsize//512+1,1)


        #--------------------------------------------------------
        # Arguments
        #--------------------------------------------------------
        self.argtype_dict = argtype_dict = { \
                'i': dict([(ct,lambda a: np.int32(a)) for ct in code_types]), \
                'd': dict([(ct,lambda a: np.float64(a)) for ct in code_types]), \
                'o': {'f90': lambda a: a.data, \
                      'c'  : lambda a: a.data, \
                      'cu' : lambda a: a.data_cu, \
                      'cl' : lambda a: a.data_cl}
                }


        # classify the argument types
        self.preset_atypes = preset_atypes = list()
        self.require_atypes = list()

        for atype in arg_types:
            assert atype in ['i','d','o','I','D','O'], "The arg_type '%s' is undefined."%(atype)

            if atype.islower():
                self.preset_atypes.append( atype )
            else:
                self.require_atypes.append( atype.lower() )


        # set the preset_args
        assert len(preset_atypes) == len(args), 'len(preset_atypes)=%d is not same as len(args)=%d'%(len(preset_atypes),len(args))

        self.preset_args = list()

        for seq, (atype, arg) in enumerate( zip(preset_atypes, args) ):
            if atype == 'o': assert arg.__class__.__name__ in ['Array','ArrayAs'], "The %d-th arguemnt is not a Array or ArrayAs instance."%(seq+1)
            self.preset_args.append( argtype_dict[atype][code_type](arg) )




    def prepared_call(self, *args):
        code_type = self.platform.code_type
        func = self.func
        argtype_dict = self.argtype_dict
        preset_atypes = self.preset_atypes
        require_atypes = self.require_atypes
        run_args = self.preset_args[:]      # copy
        

        #--------------------------------------------------------
        # Setup arguments
        #--------------------------------------------------------
        assert len(require_atypes) == len(args), 'len(require_atypes)=%d is not same as len(args)=%d'%(len(require_atypes),len(args))

        for seq, (atype, arg) in enumerate( zip(require_atypes, args) ):
            if atype == 'o': assert arg.__class__.__name__ in ['Array','ArrayAs'], "The %d-th arguemnt is not a Array or ArrayAs instance."%(len(preset_atypes)+seq+1)
            run_args.append( argtype_dict[atype][code_type](arg) )


        #--------------------------------------------------------
        # Call the prepared function
        #--------------------------------------------------------
        if code_type in ['f90', 'c']:
            func(*run_args)

        elif code_type == 'cu':
            func(*run_args, block=self.block, grid=self.grid)

        elif code_type == 'cl':
            func(self.platform.queue, (self.gsize,), None, *run_args)
