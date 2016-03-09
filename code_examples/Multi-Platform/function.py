#------------------------------------------------------------------------------
# filename  : function.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.10.28   split from machine_platform.py
#
#
# description:
#   Generalized interface to call functions from various code languages
#
# support code languages:
#   Fortran 90/95
#   C
#   CUDA
#   OpenCL
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np

from log import logger




class Function(object):
    def __init__(self, func):
        self.func = func
        
        # rules to cast arguments, it will be overrided
        self.cast_dict = { \
                'i': lambda a: np.int32(a), \
                'd': lambda a: np.float64(a) }



    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)



    def preset_arguments(self, arg_types, *args, **kwargs):
        '''
        arg_types :
            i,I: np.int32
            d,D: np.float64
            o,O: numpy array

        Lowercase letter means that it can be set before function call.
        Uppercase letter means that it should set when function call.

        # kwargs
        gsize : global thread size for CUDA and OpenCL
        '''

        # classify the argument types
        self.preset_atypes = preset_atypes = list()
        self.require_atypes = list()

        for atype in arg_types:
            if atype not in ('i','d','o','I','D','O'):
                logger.error("Error: The arg_type '%s' is not defined."%(atype))
                raise SystemExit

            if atype.islower():
                self.preset_atypes.append( atype )
            else:
                self.require_atypes.append( atype.lower() )

        # set the preset_args
        assert len(preset_atypes) == len(args), 'len(preset_atypes)=%d is not same as len(args)=%d'%(len(preset_atypes),len(args))

        self.preset_args = list()
        for seq, (atype, arg) in enumerate( zip(preset_atypes, args) ):
            if atype=='o' and arg.__class__.__name__ not in ['Array','ArrayAs']:
                logger.error("Error: The %d-th arguemnt is not a Array or ArrayAs instance."%(seq+1))
                raise SystemExit
            self.preset_args.append( self.cast_dict[atype](arg) )



    def get_casted_arguments(self, *args):
        if len(self.require_atypes) != len(args):
            logger.error('Error: len(require_atypes)=%d is not same as len(args)=%d'%(len(self.require_atypes),len(args)))
            raise SystemExit

        casted_args = self.preset_args[:]  # copy
        for seq, (atype, arg) in enumerate( zip(self.require_atypes, args) ):
            if arg.__class__.__name__ not in ['Array','ArrayAs']:
                logger.error("Error: The %d-th arguemnt is not a Array or ArrayAs instance."%(seq+1))
                raise SystemExit
            casted_args.append( self.cast_dict[atype](arg) )

        return casted_args



    def get_gsize(self, **kwargs):
        try:
            gsize = kwargs['gsize']
        except Exception, e:
            logger.error("Error: When the code_type is 'cu' or 'cl', the 'gsize' keyward argument must be specified.", exc_info=True)
            raise SystemExit

        return gsize




class Function_F90_C(Function):
    def __init__(self, func):
        super(Function_F90_C, self).__init__(func)


    def prepare(self, arg_types, *args, **kwargs):
        self.cast_dict['o'] = lambda a: a.data
        self.preset_arguments(arg_types, *args, **kwargs)


    def prepared_call(self, *args):
        casted_args = self.get_casted_arguments(*args)
        self.func(*casted_args)




class Function_CUDA(Function):
    def __init__(self, func):
        super(Function_CUDA, self).__init__(func)


    def prepare(self, arg_types, *args, **kwargs):
        gsize = self.get_gsize(**kwargs)
        self.block = (512,1,1)
        self.grid = (gsize//512+1,1)

        self.cast_dict['o'] = lambda a: a.data_cu
        self.preset_arguments(arg_types, *args, **kwargs)


    def prepared_call(self, *args):
        casted_args = self.get_casted_arguments(*args)
        self.func(*casted_args, block=self.block, grid=self.grid)




class Function_OpenCL(Function):
    def __init__(self, queue, func):
        super(Function_OpenCL, self).__init__(func)
        self.queue = queue


    def prepare(self, arg_types, *args, **kwargs):
        self.gsize = self.get_gsize(**kwargs)
        self.cast_dict['o'] = lambda a: a.data_cl
        self.preset_arguments(arg_types, *args, **kwargs)


    def prepared_call(self, *args):
        casted_args = self.get_casted_arguments(*args)
        self.func(self.queue, (self.gsize,), None, *casted_args)
