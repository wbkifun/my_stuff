import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

import io
import yaml
import sys
from os.path import abspath, dirname, join
current_dpath = dirname(abspath(__file__))
sys.path.append(current_dpath)
from device_platform import DevicePlatform



def capture(func):
    def wrapper(*args, **kwargs):
        capturer1 = io.StringIO()
        capturer2 = io.StringIO()
        old_stdout, sys.stdout = sys.stdout, capturer1
        old_stderr, sys.stderr = sys.stderr, capturer2

        ret = func(*args, **kwargs)

        sys.stdout, sys.stderr = old_stdout, old_stderr
        out = capturer1.getvalue().rstrip('\n')
        err = capturer2.getvalue().rstrip('\n')

        return ret, out, err

    return wrapper




def check_apb_amb(platform):
    base_dpath = join(current_dpath, 'src')

    #
    # Device platform
    #
    ret, out, err = capture(platform.clean_modules)(base_dpath)
    build_dpath, out, err = capture(platform.build_modules)(base_dpath, generate_header=True)

    lib_apb, out, err = capture(platform.load_module)(build_dpath, 'apb')
    lib_amb, out, err = capture(platform.load_module)(build_dpath, 'amb')
    apb = platform.get_function(lib_apb, 'apb')
    amb = platform.get_function(lib_amb, 'amb')

    #
    # Setup
    #
    nx = 1000000
    a = np.random.rand(nx)
    b = np.random.rand(nx)
    c = np.random.rand(nx)
    c2 = c.copy()

    with open(join(base_dpath, 'apb.yaml'), 'r') as f: apb_dict = yaml.load(f)
    with open(join(base_dpath, 'amb.yaml'), 'r') as f: amb_dict = yaml.load(f)
    kk, lll, mm = apb_dict['kk'], apb_dict['lll'], amb_dict['section']['mm']

    ref = kk*a + lll*b + mm*c

    #
    # Device arrays
    #
    a_dev = platform.ArrayAs(a)
    b_dev = platform.ArrayAs(b)
    c_dev = platform.ArrayAs(c)
    c2_dev = platform.ArrayAs(c2)

    #
    # Verify
    #
    apb.prepare('iooo', nx, a_dev, b_dev, c_dev, gsize=nx)
    apb.prepared_call()
    aa_equal(ref, c_dev.get(), 14)

    amb.prepare('iooo', nx, a_dev, b_dev, c2_dev, gsize=nx)
    amb.prepared_call()
    aa_equal(ref, c2_dev.get(), 14)




def test_cpu_f90():
    '''
    DevicePlatform with build: CPU, f90
    '''
    platform = DevicePlatform('CPU', 'f90')
    check_apb_amb(platform)




def test_cpu_c():
    '''
    DevicePlatform with build: CPU, c
    '''
    platform = DevicePlatform('CPU', 'C')
    check_apb_amb(platform)




def test_cpu_opencl():
    '''
    DevicePlatform with build: CPU, OpenCL
    '''
    platform = DevicePlatform('CPU', 'OPENCL', vendor_name='Intel')
    check_apb_amb(platform)



def test_nvidia_gpu_cuda():
    '''
    DevicePlatform with build: NVIDIA_GPU, CUDA
    '''
    platform = DevicePlatform('NVIDIA_GPU', 'CUDA')
    check_apb_amb(platform)




def test_nvidia_gpu_opencl():
    '''
    DevicePlatform with build: NVIDIA_GPU, OpenCL
    '''
    platform = DevicePlatform('NVIDIA_GPU', 'OPENCL')
    check_apb_amb(platform)




#==============================================================================
# Test PyKIM fail cases
#==============================================================================

def check_set12(platform):
    base_dpath = join(current_dpath, 'src12')

    #
    # Device platform
    #
    ret, out, err = capture(platform.clean_modules)(base_dpath)
    build_dpath, out, err = capture(platform.build_modules)(base_dpath, generate_header=True)

    lib, out, err = capture(platform.load_module)(build_dpath, 'set12')
    apb = platform.get_function(lib, 'calc_divv')

    #
    # Setup
    #
    ngq, nlev, nelem = 4, 50, 5400
    ref = np.ones(ngq*ngq*(nlev+1)*nelem, 'f8')*1.2

    #
    # Device arrays
    #
    ru = platform.Array((ngq,ngq,nlev+1,nelem), 'f8')

    #
    # Verify
    #
    size3d = ngq*ngq*(nlev+1)*nelem
    apb.prepare('iiio', ngq, nlev, nelem, ru, gsize=size3d)
    apb.prepared_call()
    if platform.code_type == 'f90':
        out = ru.get().ravel(order='F')
    else:
        out = ru.get()
    a_equal(out, ref)




def test_set12_cpu_f90():
    '''
    DevicePlatform with build, set1.2: CPU, f90
    '''
    platform = DevicePlatform('CPU', 'f90')
    check_set12(platform)




def test_set12_cpu_c():
    '''
    DevicePlatform with build, set1.2: CPU, c
    '''
    platform = DevicePlatform('CPU', 'C')
    check_set12(platform)




def test_set12_cpu_opencl():
    '''
    DevicePlatform with build, set1.2: CPU, OpenCL
    '''
    platform = DevicePlatform('CPU', 'OpenCL', vendor_name='Intel')
    check_set12(platform)




def test_set12_nvidia_gpu_cuda():
    '''
    DevicePlatform with build, set1.2: NVIDIA_GPU, CUDA
    '''
    platform = DevicePlatform('NVIDIA_GPU', 'CUDA')
    check_set12(platform)




def test_set12_nvidia_gpu_opencl():
    '''
    DevicePlatform with build, set1.2: NVIDIA_GPU, OpenCL
    '''
    platform = DevicePlatform('NVIDIA_GPU', 'OpenCL')
    check_set12(platform)
