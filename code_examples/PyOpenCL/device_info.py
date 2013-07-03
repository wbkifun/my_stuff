from __future__ import division
import pyopencl as cl


platforms = cl.get_platforms()

for i, platform in enumerate(platforms):
    print('\n'+'-'*47)
    print('Platform #%d' % i)
    print('  name: %s' % platform.name)
    print('  vendor: %s' % platform.vendor)
    print('  version: %s' % platform.version)
    print('')

    devices = platform.get_devices()
    
    for j, device in enumerate(devices):
        print('    Device #%d' % j)
        print('\tname: %s' % device.name)
        print('\tmax compute unis: %d' % device.max_compute_units)
        print('\tglobal mem size: %d' % device.global_mem_size)
        print('\tlocal mem size: %d' % device.local_mem_size)
        print('\tmax work-group size: %d' % device.max_work_group_size)
        print('\tmax work-item sizes: %s' % device.max_work_item_sizes)
