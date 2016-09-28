#------------------------------------------------------------------------------
# filename  : function.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.10.28   split from machine_platform.py
#             2016.3.10    add traceback.format_stack from log
#             2016.3.11    split a kernel when the grid size is over 65535
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

from log import logger, get_stack




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
                logger.error("The arg_type '%s' is not defined.\n{}".format(atype,get_stack()))
                raise SystemExit

            if atype.islower():
                self.preset_atypes.append( atype )
            else:
                self.require_atypes.append( atype.lower() )

        # set the preset_args
        if not len(preset_atypes) == len(args):
            logger.error("len(preset_atypes)={} is not same as len(args)={}\n{}".format(len(preset_atypes), len(args), get_stack()))

        self.preset_args = list()
        for seq, (atype, arg) in enumerate( zip(preset_atypes, args) ):
            if atype=='o':
                assert arg.__class__.__name__ in ['Array','ArrayAs'], "The {}-th arguemnt is not a Array or ArrayAs instance.".format(seq)
            self.preset_args.append( self.cast_dict[atype](arg) )



    def get_casted_arguments(self, *args):
        if len(self.require_atypes) != len(args):
            logger.error("len(require_atypes)={} is not same as len(args)={}\n{}".format(len(self.require_atypes), len(args), get_stack()))
            raise SystemExit

        casted_args = self.preset_args[:]  # copy
        for seq, (atype, arg) in enumerate( zip(self.require_atypes, args) ):
            if atype == 'O' and \
               arg.__class__.__name__ not in ['Array','ArrayAs']:
                logger.error("The {}-th arguemnt is not a Array or ArrayAs instance.\n{}".format(seq+1, get_stack()) )
                raise SystemExit
            casted_args.append( self.cast_dict[atype](arg) )

        return casted_args



    def get_gsize(self, **kwargs):
        assert 'gsize' in kwargs.keys(), "When the code_type is 'cu' or 'cl', the 'gsize' keyward argument must be specified."

        return kwargs['gsize']




class Function_F90_C(Function):
    def __init__(self, func):
        super(Function_F90_C, self).__init__(func)


    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)


    def prepare(self, arg_types, *args, **kwargs):
        self.cast_dict['o'] = lambda a: a.data
        self.preset_arguments(arg_types, *args, **kwargs)


    def prepared_call(self, *args):
        casted_args = self.get_casted_arguments(*args)
        self.func(*casted_args)




class Function_CUDA(Function):
    def __init__(self, func):
        super(Function_CUDA, self).__init__(func)


    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)


    def prepare(self, arg_types, *args, **kwargs):
        # thread, grid size
        gsize = self.get_gsize(**kwargs)
        thread_per_block = kwargs.get('thread_per_block', 512)
        block_per_grid = int(gsize)//thread_per_block + 1

        max_block = 65535
        niter = block_per_grid//max_block + 1
        remain_block = block_per_grid - max_block*(niter-1)

        block = (thread_per_block,1,1)
        grids = [(max_block,1) for i in range(niter-1)] + [(remain_block,1)]

        #print("gsize={}, block={}, grids={}".format(gsize, block, grids))
        self.thread_per_block = thread_per_block
        self.block_per_grid = block_per_grid
        self.max_block = max_block
        self.block = block
        self.grids = grids

        # preset arguments
        self.cast_dict['o'] = lambda a: a.data_cu
        self.preset_arguments(arg_types, *args, **kwargs)


    def prepared_call(self, *args):
        casted_args = self.get_casted_arguments(*args)

        for i, grid in enumerate(self.grids):
            shift_gid = np.int32( i*self.max_block*self.thread_per_block )
            casted_args2 = [shift_gid] + casted_args
            self.func(*casted_args2, block=self.block, grid=grid)




class Function_OpenCL(Function):
    def __init__(self, queue, func):
        super(Function_OpenCL, self).__init__(func)
        self.queue = queue


    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)


    def prepare(self, arg_types, *args, **kwargs):
        self.gsize = self.get_gsize(**kwargs)
        self.work_group_size = kwargs.get('work_group_size', 128)

        self.cast_dict['o'] = lambda a: a.data_cl
        self.preset_arguments(arg_types, *args, **kwargs)


    def prepared_call(self, *args):
        casted_args = self.get_casted_arguments(*args)
        self.func(self.queue, (self.gsize,), (self.work_group_size,), *casted_args)
