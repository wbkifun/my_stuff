#------------------------------------------------------------------------------
# filename  : nvidia_gpu_info.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.29    start
#------------------------------------------------------------------------------

import pycuda.driver as cuda



cuda.init()

gpu_dict = dict()
for i in xrange(cuda.Device.count()):
    device = cuda.Device(i)
    gpu_dict[device.name()] = device


for name, device in gpu_dict.items():
    print 'NVIDIA %s (%d Devices)'%(name, device.count())  
    print '\tTotal memory: %d'%(device.total_memory())

    attrs = device.get_attributes()
    for key, val in attrs.iteritems():
        print '\t%s: %s'%(key, str(val))
