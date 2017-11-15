'''

abstarct : Use the hand-written C library using numpy

history :
  2015-10-29  Ki-Hwan Kim  start
  2017-09-04  Ki-Hwan Kim  Python3

'''

from __future__ import print_function, division
import pycuda.driver as cuda


cuda.init()

gpu_dict = dict()
for i in xrange(cuda.Device.count()):
    device = cuda.Device(i)
    gpu_dict[device.name()] = device


for name, device in gpu_dict.items():
    print('NVIDIA {} ({} Devices)'.format(name, device.count()))
    print('\tTotal memory: {}'.format(device.total_memory()))

    attrs = device.get_attributes()
    for key, val in attrs.iteritems():
        print('\t{}: {}'.format(key, str(val)))
